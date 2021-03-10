#include "hcmlabpupiltracker.h"

#include <iostream>

#include "util/hcmutils.h"
#include "outputwriters/hcmlabpupildatacsvwriter.h"
#include "outputwriters/hcmlabpupildatassiwriter.h"


HCMLabPupilTracker::HCMLabPupilTracker(int inputWidth, int inputHeight, double inputfps, bool exportSSIStream,
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
              "_DEBUG.mp4"),
      m_debugOutputSize(
          cv::Size(inputWidth / 3,
                   inputHeight / 3 * 2))
{
    m_debugOutputMat = cv::Mat::zeros(m_debugOutputSize, CV_8UC3);

    if (m_exportCSV) {
        outputWriters.push_back(std::make_unique<HCMLabPupilDataCSVWriter>(outputDirPath, outputBaseName));
    }

    if (m_exportSSIStream) {
        outputWriters.push_back(std::make_unique<HCMLabPupilDataSSIWriter>(outputDirPath, outputBaseName, inputfps));
    }
}

bool HCMLabPupilTracker::init()
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

PupilTrackingDataFrame HCMLabPupilTracker::process(const cv::Mat &inputFrame,
                                                   size_t frameNr)
{
    m_eyeExtractor.process(inputFrame, frameNr, m_rightEyeMat, m_leftEyeMat);

    PupilData leftPupilData, rightPupilData;
    if (m_renderDebugVideo) {
        leftPupilData = m_detectorLeft.process(m_leftEyeMat, m_leftDebugMat);
        rightPupilData = m_detectorRight.process(m_rightEyeMat, m_rightDebugMat);
    } else {
        leftPupilData = m_detectorLeft.process(m_leftEyeMat);
        rightPupilData = m_detectorRight.process(m_rightEyeMat);
    }

    PupilTrackingDataFrame trackingData = {leftPupilData, rightPupilData};

    m_trackingData.push_back(trackingData);

    if (m_renderDebugVideo) {
        writeDebugFrame();
    }

    return trackingData;
}

bool HCMLabPupilTracker::stop()
{
    bool retVal = true;
    if (!m_eyeExtractor.stop().ok()) {
        hcmutils::logError("Problem stopping the iristracking mediapipe graph");
        retVal = false;
    }

    writeOutTrackingData();

    return retVal;
}

void HCMLabPupilTracker::writeOutTrackingData()
{
    for (const auto &writer : outputWriters) {
        writer->write(m_trackingData);
    }
}

void HCMLabPupilTracker::writeToDebugFrame(const cv::Mat& input, int targetX, int targetY, int maxWidth, int maxHeight)
{
    //determine scaling factor to fit the input image inside the maximum possible area defined by maxWidth and maxHeight
    auto scaleFactorX = maxWidth / (input.cols * 1.0);
    auto scaleFactorY = maxHeight / (input.rows * 1.0);

    auto scaleFactor = std::min(scaleFactorX, scaleFactorY);

    cv::Mat inputResized;
    cv::resize(input, inputResized, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    cv::Rect targetRoi(targetX, targetY, inputResized.cols, inputResized.rows);

    auto targetMat = m_debugOutputMat(targetRoi);

    inputResized.copyTo(targetMat);
}

/***
 * Renders debug information into an image:
 *
 * ------------      --------------
 * |          |      |            |
 * | Left Eye |      | Right Eye  |
 * | (source) |      | (source)   |
 * |          |      |            |
 * ------------      --------------
 *
 * ------------      --------------
 * |          |      |            |
 * | Left Eye |      |  Right Eye |
 * |(tracking)|      | (tracking) |
 * |          |      |            |
 * ------------      --------------
 */
void HCMLabPupilTracker::writeDebugFrame()
{
    m_debugOutputMat = cv::Scalar(0, 0, 0);

    int maxWidth = (m_debugOutputSize.width - 3 * m_debugPadding) / 2;
    int maxHeight = (m_debugOutputSize.height - 3 * m_debugPadding) / 2;

    auto const leftColX = m_debugPadding;
    auto const rightColX = m_debugPadding + maxWidth + m_debugPadding;

    auto firstRowY = m_debugPadding;
    auto secondRowY = m_debugPadding + maxHeight;

    //left normal eye
    writeToDebugFrame(m_leftEyeMat, leftColX, firstRowY, maxWidth, maxHeight);

    // left tracking output
    writeToDebugFrame(m_leftDebugMat, leftColX, secondRowY, maxWidth, maxHeight);

    // right normal eye
    writeToDebugFrame(m_rightEyeMat, rightColX, firstRowY, maxWidth, maxHeight);

    // right tracking output
    writeToDebugFrame(m_rightDebugMat, rightColX, secondRowY, maxWidth, maxHeight);


    cv::cvtColor(m_debugOutputMat, m_debugOutputMat, cv::COLOR_RGB2BGR);
    m_debugVideoWriter.write(m_debugOutputMat);
}