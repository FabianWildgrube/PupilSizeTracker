#include "hcmlabfullfacepupiltracker.h"

#include <iostream>

#include "util/hcmutils.h"
#include "outputwriters/hcmlabpupildatacsvwriter.h"
#include "outputwriters/hcmlabpupildatassiwriter.h"


HCMLabFullFacePupilTracker::HCMLabFullFacePupilTracker(int inputWidth, int inputHeight, double inputfps, bool exportSSIStream,
                                       bool exportCSV, bool renderDebugVideo, std::string outputDirPath,
                                       std::string outputBaseName)
    : m_inputWidth(inputWidth),
      m_inputHeight(inputHeight),
      m_fps(inputfps),
      m_exportSSIStream(exportSSIStream),
      m_exportCSV(exportCSV),
      m_renderDebugVideo(renderDebugVideo),
      m_debugVideoOutputPath(
          outputDirPath + outputBaseName +
              "_TRACKED_VIDEO.mp4"),
      m_eyeExtractor(HCMLabEyeExtractor(inputfps))
{
    int debugOutputWidth = m_debugPadding
                            + inputWidth / m_debugSourceVideoScaleDivider
                            + m_debugPadding
                            + m_debugVideoEyeSize
                            + m_debugPadding
                            + m_debugVideoEyeSize
                            + m_debugPadding;
    
    int debugOutputHeight = m_debugPadding
                            + std::max(inputHeight / m_debugSourceVideoScaleDivider, 2 * m_debugVideoEyeSize + m_debugPadding)
                            + m_debugPadding;

    m_debugOutputSize = cv::Size(debugOutputWidth, debugOutputHeight);

    m_debugOutputMat = cv::Mat::zeros(m_debugOutputSize, CV_8UC3);

    if (m_exportCSV) {
        m_outputWriters.push_back(std::make_unique<HCMLabPupilDataCSVWriter>(outputDirPath, outputBaseName));
    }

    if (m_exportSSIStream) {
        m_outputWriters.push_back(std::make_unique<HCMLabPupilDataSSIWriter>(outputDirPath, outputBaseName, inputfps));
    }
}

bool HCMLabFullFacePupilTracker::init()
{
    if (!m_eyeExtractor.init().ok()) {
        hcmutils::logError("Could not init Eye extractor");
        return false;
    }
    hcmutils::logInfo("Initialized EyeExtractor");

    if (m_renderDebugVideo) {
        m_debugVideoWriter.open(m_debugVideoOutputPath, mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                                m_fps, m_debugOutputSize);

        if (!m_debugVideoWriter.isOpened()) {
            hcmutils::logError("Debug Videowriter could not be opened with path: " + m_debugVideoOutputPath);
            return false;
        }
        hcmutils::logInfo("Initialized Debug Videowriter");
    }

    return true;
}

PupilTrackingDataFrame HCMLabFullFacePupilTracker::process(const cv::Mat &inputFrame,
                                                   size_t frameNr)
{
    IrisDiameters irisDiameters = m_eyeExtractor.process(inputFrame, frameNr, m_rightEyeMat, m_leftEyeMat);

    RawPupilData leftPupilDataRaw, rightPupilDataRaw;
    if (m_renderDebugVideo) {
        leftPupilDataRaw = m_detectorLeft.process(m_leftEyeMat, m_leftDebugMat);
        rightPupilDataRaw = m_detectorRight.process(m_rightEyeMat, m_rightDebugMat);
    } else {
        leftPupilDataRaw = m_detectorLeft.process(m_leftEyeMat);
        rightPupilDataRaw = m_detectorRight.process(m_rightEyeMat);
    }

    PupilTrackingDataFrame trackingData = {PupilData(leftPupilDataRaw, irisDiameters.left), PupilData(rightPupilDataRaw, irisDiameters.right)};

    m_trackingData.push_back(trackingData);

    if (m_renderDebugVideo) {
        writeDebugFrame(inputFrame);
    }

    return trackingData;
}

bool HCMLabFullFacePupilTracker::stop()
{
    bool retVal = true;

    if (m_debugVideoWriter.isOpened()) {
        m_debugVideoWriter.release();
    }

    if (!m_eyeExtractor.stop().ok()) {
        hcmutils::logError("Problem stopping the iristracking mediapipe graph");
        retVal = false;
    }

    writeOutTrackingData();

    return retVal;
}

void HCMLabFullFacePupilTracker::writeOutTrackingData()
{
    for (const auto &writer : m_outputWriters) {
        writer->write(m_trackingData);
    }
}

/***
 * Renders debug information into an image:
 *
 *                            ------------   --------------
 *                            |          |   |            |
 *                            | Left Eye |   | Right Eye  |
 * ------------------------   | (source) |   | (source)   |
 * |                       |  |          |   |            |
 * |                       |  ------------   --------------
 * |     source video      |
 * |                       |  ------------   --------------
 * |                       |  |          |   |            |
 * |                       |  | Left Eye |   |  Right Eye |
 * ------------------------   |(tracking)|   | (tracking) |
 *                            |          |   |            |
 *                            ------------   --------------
 */
void HCMLabFullFacePupilTracker::writeDebugFrame(const cv::Mat &inputFrame)
{
    m_debugOutputMat = cv::Scalar(0, 0, 0);

    int sourceVideoScaledWidth = inputFrame.cols / m_debugSourceVideoScaleDivider;
    int sourceVideoScaledHeight = inputFrame.rows / m_debugSourceVideoScaleDivider;

    int sourceVideoY = (m_debugOutputSize.height - sourceVideoScaledHeight) / 2; //center source video vertically

    auto const leftColX = m_debugPadding + sourceVideoScaledWidth + m_debugPadding;
    auto const rightColX = leftColX + m_debugVideoEyeSize + m_debugPadding;

    auto firstRowY = m_debugPadding;
    auto secondRowY = firstRowY + m_debugVideoEyeSize + m_debugPadding;

    //source video
    hcmutils::writeIntoFrame(m_debugOutputMat, inputFrame, m_debugPadding, sourceVideoY, sourceVideoScaledWidth, sourceVideoScaledHeight);

    //left normal eye
    hcmutils::writeIntoFrame(m_debugOutputMat, m_leftEyeMat, leftColX, firstRowY, m_debugVideoEyeSize, m_debugVideoEyeSize);

    // left tracking output
    hcmutils::writeIntoFrame(m_debugOutputMat, m_leftDebugMat, leftColX, secondRowY, m_debugVideoEyeSize, m_debugVideoEyeSize);

    // right normal eye
    hcmutils::writeIntoFrame(m_debugOutputMat, m_rightEyeMat, rightColX, firstRowY, m_debugVideoEyeSize, m_debugVideoEyeSize);

    // right tracking output
    hcmutils::writeIntoFrame(m_debugOutputMat, m_rightDebugMat, rightColX, secondRowY, m_debugVideoEyeSize, m_debugVideoEyeSize);


    cv::cvtColor(m_debugOutputMat, m_debugOutputMat, cv::COLOR_RGB2BGR);
    m_debugVideoWriter.write(m_debugOutputMat);
}