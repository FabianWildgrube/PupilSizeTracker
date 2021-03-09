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

int HCMLabPupilTracker::writeToDebugFrame(const cv::Mat& input, int targetX, int targetY)
{
    auto safeWidth = std::min(input.cols, (m_debugOutputSize.width - 3 * m_debugPadding) / 2);
    auto safeHeight = std::min(input.rows, (m_debugOutputSize.height - 3 * m_debugPadding) / 2);

    cv::Rect sourceRoi(0, 0, safeWidth, safeHeight); // this only crops. Scale, if you want all the info

    cv::Rect targetRoi(targetX, targetY, safeWidth, safeHeight);

    auto targetMat = m_debugOutputMat(targetRoi);

    input(sourceRoi).copyTo(targetMat);

    return safeHeight;
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

    auto const leftColX = m_debugPadding;
    auto const rightColX = m_debugPadding + ((m_debugOutputSize.width - 3 * m_debugPadding) / 2) + m_debugPadding;

    auto leftColCurrentY = m_debugPadding;
    auto rightColCurrentY = m_debugPadding;

    //left normal eye
    auto writtenHeight = writeToDebugFrame(m_leftEyeMat, leftColX, leftColCurrentY);
    leftColCurrentY += writtenHeight + m_debugPadding;

    // left tracking output
    writeToDebugFrame(m_leftDebugMat, leftColX, leftColCurrentY);

    // right normal eye
    writtenHeight = writeToDebugFrame(m_rightEyeMat, rightColX, rightColCurrentY);
    rightColCurrentY += writtenHeight + m_debugPadding;

    // right tracking output
    writeToDebugFrame(m_rightDebugMat, rightColX, rightColCurrentY);


    cv::cvtColor(m_debugOutputMat, m_debugOutputMat, cv::COLOR_RGB2BGR);
    m_debugVideoWriter.write(m_debugOutputMat);
}