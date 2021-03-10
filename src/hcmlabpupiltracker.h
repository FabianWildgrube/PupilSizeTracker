#ifndef HCMLAB_PUPILTRACKER_H
#define HCMLAB_PUPILTRACKER_H

#include <string>
#include <vector>
#include <memory>

#include "util/hcmdatatypes.h"
#include "hcmlabeyeextractor.h"
#include "hcmlabpupildetector.h"
#include "outputwriters/hcmlabpupildataoutputwriter.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

class HCMLabPupilTracker
{
public:
    HCMLabPupilTracker(int inputWidth, int inputHeight, double inputfps, bool exportSSIStream,
                       bool exportCSV, bool renderDebugVideo, std::string outputDirPath,
                       std::string outputBaseName);

    ~HCMLabPupilTracker()
    {};

    bool init();

    /// Tracks human pupils and their size in the given inputFrame. Meant for online use (i.e. call this function for each frame of a stream of frames).
    /// @param inputFrame - a single frame of the input video containing a human face
    /// @param frameNr - number of the frame within the source video, used as a timecode
    PupilTrackingDataFrame process(const cv::Mat &inputFrame, size_t frameNr);

    bool stop();

private:
    void writeDebugFrame(const cv::Mat &inputFrame);

    void writeOutTrackingData();

private:
    void writeToDebugFrame(const cv::Mat& input, int targetX, int targetY, int maxWidth, int maxHeight);

    HCMLabEyeExtractor m_eyeExtractor;
    HCMLabPupilDetector m_detectorLeft;
    HCMLabPupilDetector m_detectorRight;

    cv::Mat m_leftEyeMat, m_rightEyeMat;

    std::vector<PupilTrackingDataFrame> m_trackingData;

    std::vector<std::unique_ptr<HCMLabPupilDataOutputWriter_I>> outputWriters;

    int m_inputWidth, m_inputHeight;
    double m_fps;

    bool m_exportCSV;
    bool m_exportSSIStream;

    bool m_renderDebugVideo;
    std::string m_debugVideoOutputPath;
    cv::Size m_debugOutputSize;
    cv::VideoWriter m_debugVideoWriter;
    cv::Mat m_leftDebugMat, m_rightDebugMat, m_debugOutputMat;
    int m_debugPadding = 10;
    int m_debugSourceVideoScaleDivider = 3; // the inverse of this will be used to scale down the whokle input video in the debug video output
    int m_debugVideoEyeSize = 300; // the side length in pixels to which the eye crops will be rendered in the debug video
};

#endif // HCMLAB_PUPILTRACKER_H