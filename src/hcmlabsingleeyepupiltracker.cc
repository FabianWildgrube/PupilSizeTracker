#include "hcmlabsingleeyepupiltracker.h"

#include <iostream>

#include "util/hcmutils.h"
#include "outputwriters/hcmlabpupildatacsvwriter.h"
#include "outputwriters/hcmlabpupildatassiwriter.h"


HCMLabSingleEyePupilTracker::HCMLabSingleEyePupilTracker(int inputWidth, int inputHeight, double inputfps, bool exportSSIStream,
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
              "_TRACKED_VIDEO.mp4")
{
    int debugOutputWidth = m_debugPadding
                            + m_debugVideoEyeSize
                            + m_debugPadding
                            + m_debugVideoEyeSize
                            + m_debugPadding;
    
    int debugOutputHeight = m_debugPadding
                            + m_debugVideoEyeSize
                            + m_debugPadding;

    m_debugOutputSize = cv::Size(debugOutputWidth, debugOutputHeight);

    m_debugOutputMat = cv::Mat::zeros(m_debugOutputSize, CV_8UC3);

    std::cout << "Width: " << m_inputWidth << ", Height: " << m_inputHeight << ", fps: " << m_fps << ", outputpath: " << m_debugVideoOutputPath << "\n";
    std::cout << "Debug Render: " << (m_renderDebugVideo ? "true" : "false") << "\n";
    std::cout << "Debug output mat: " << m_debugOutputMat.cols << ", " << m_debugOutputMat.rows << "\n";

    if (m_exportCSV) {
        m_outputWriters.push_back(std::make_unique<HCMLabPupilDataCSVWriter>(outputDirPath, outputBaseName));
    }

    if (m_exportSSIStream) {
        m_outputWriters.push_back(std::make_unique<HCMLabPupilDataSSIWriter>(outputDirPath, outputBaseName, inputfps));
    }
}

bool HCMLabSingleEyePupilTracker::init()
{
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

PupilTrackingDataFrame HCMLabSingleEyePupilTracker::process(const cv::Mat &inputFrame,
                                                   size_t frameNr)
{
    IrisDiameters irisDiameters = {1.0f, 1.0f}; //dummy diameters because footage from an eye-tracker is always constant distance from the eye

    RawPupilData pupilDataRaw;
    if (m_renderDebugVideo) {
        pupilDataRaw = m_pupilDetector.process(inputFrame, m_eyeDebugMat);
    } else {
        pupilDataRaw = m_pupilDetector.process(inputFrame);
    }

    //duplicate tracking data to adhere to data format that was designed for tracking two eyes!
    PupilTrackingDataFrame trackingData = {PupilData(pupilDataRaw, irisDiameters.left), PupilData(pupilDataRaw, irisDiameters.left)};

    m_trackingData.push_back(trackingData);

    if (m_renderDebugVideo) {
        writeDebugFrame(inputFrame);
    }

    return trackingData;
}

bool HCMLabSingleEyePupilTracker::stop()
{
    if (m_debugVideoWriter.isOpened()) {
        m_debugVideoWriter.release();
    }

    writeOutTrackingData();
    return true;
}

void HCMLabSingleEyePupilTracker::writeOutTrackingData()
{
    for (const auto &writer : m_outputWriters) {
        writer->write(m_trackingData);
    }
}

/***
 * Renders debug information into an image:
 *
 *  ------------   --------------
 *  |          |   |            |
 *  |          |   |            |
 *  | (source) |   |  (debug)   |
 *  |          |   |            |
 *  ------------   --------------
 */
void HCMLabSingleEyePupilTracker::writeDebugFrame(const cv::Mat &inputFrame)
{
    m_debugOutputMat = cv::Scalar(0, 0, 0);

    int videoY = (m_debugOutputSize.height - m_debugVideoEyeSize) / 2; //center source video vertically

    auto const rightColX = m_debugPadding + m_debugVideoEyeSize + m_debugPadding;

    //source video
    hcmutils::writeIntoFrame(m_debugOutputMat, inputFrame, m_debugPadding, videoY, m_debugVideoEyeSize, m_debugVideoEyeSize);

    // left tracking output
    hcmutils::writeIntoFrame(m_debugOutputMat, m_eyeDebugMat, rightColX, videoY, m_debugVideoEyeSize, m_debugVideoEyeSize);

    cv::cvtColor(m_debugOutputMat, m_debugOutputMat, cv::COLOR_RGB2BGR);
    m_debugVideoWriter.write(m_debugOutputMat);
}