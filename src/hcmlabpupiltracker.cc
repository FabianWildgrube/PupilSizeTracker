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
          cv::Size(inputWidth / 2,
                   inputHeight / 2))
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
        std::cout << "Outputsize: " << m_debugOutputSize << ", fps: " << m_fps << "\n";
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
    std::cout << "Before Eyeextractor\n";
    m_eyeExtractor.process(inputFrame, frameNr, m_rightEyeMat, m_leftEyeMat);
    std::cout << "After Eyeextractor\n";

    PupilData leftPupilData, rightPupilData;
    if (m_renderDebugVideo) {
        leftPupilData = m_detectorLeft.process(m_leftEyeMat, m_leftDebugMat);
        rightPupilData = m_detectorRight.process(m_rightEyeMat, m_rightDebugMat);
    } else {
        leftPupilData = m_detectorLeft.process(m_leftEyeMat);
        rightPupilData = m_detectorRight.process(m_rightEyeMat);
    }
    std::cout << "After Pupil Detectors\n";

    PupilTrackingDataFrame trackingData = {leftPupilData, rightPupilData};

    m_trackingData.push_back(trackingData);

    if (m_renderDebugVideo) {
        writeDebugFrame();
    }
    std::cout << "After Render Debug\n";

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

void HCMLabPupilTracker::writeDebugFrame()
{
    cv::Rect leftSourceRoi(0, 0,
                           std::min(m_leftDebugMat.cols, m_debugOutputSize.width / 2),
                           std::min(m_leftDebugMat.rows, m_debugOutputSize.height));
    cv::Rect targetRoi(20,
                       std::max((m_debugOutputMat.rows / 2) - (m_leftDebugMat.rows / 2), 0),
                       std::min(m_leftDebugMat.cols, m_debugOutputSize.width / 2),
                       std::min(m_leftDebugMat.rows, m_debugOutputSize.height));
    auto targetMat = m_debugOutputMat(targetRoi);

    m_leftDebugMat(leftSourceRoi).copyTo(targetMat);


    cv::Rect rightSourceRoi(0, 0,
                            std::min(m_rightDebugMat.cols, m_debugOutputSize.width / 2),
                            std::min(m_rightDebugMat.rows, m_debugOutputSize.height));

    cv::Rect targetRoi2(m_debugOutputSize.width / 2 - 50,
                        std::max((m_debugOutputMat.rows / 2) - (m_rightDebugMat.rows / 2), 0),
                        std::min(m_rightDebugMat.cols, m_debugOutputSize.width / 2),
                        std::min(m_rightDebugMat.rows, m_debugOutputSize.height));
    cv::Mat targetMat2 = m_debugOutputMat(targetRoi2);

    m_rightDebugMat(rightSourceRoi).copyTo(targetMat2);


    cv::cvtColor(m_debugOutputMat, m_debugOutputMat, cv::COLOR_RGB2BGR);
    m_debugVideoWriter.write(m_debugOutputMat);
}