/**
 * Pupil-Size-Tracking without head-mounted eye-tracking cameras:
 * Track the size of both of a human's pupils just from the video of their face.
 *
 * For best results, use a camera that records in the infrared spectrum (750nm and above).
 * Illuminate the eyes with an infrared light source, that preferrably is a point light, in order to keep
 * its reflection in the eyes as small as possible.
 * The picture should have good contrast, too.
 *
 * This program expects a video file containing a human face as input.
 * The video must contain both eyes as well as at least part of the nose, cheek and forehead.
 * For best results stay as close to the face as possible while honoring these requirements.
 *
 * The ouput is a .json file containing pixel coordinates for both irises in each frame
 * and a .csv file containing the pupil-diameter and confidence of that value for both eyes in each frame.
 *
 * If the "render_pupil_tracking" flag is set, two videos of the eye regions with the tracked pupil-diameter
 * and confidence rendered onto them are created as well.
 *
 * Written by Fabian Wildgrube, HCMLab 2020-2021
 *
 * Makes use of:
 *  Google's Mediapipe:
 *          Camillo Lugaresi, et al., MediaPipe: A Framework for Building Perception Pipelines,
 *          2019, https://arxiv.org/abs/1906.08172
 *
 *  PuRe and PuReSt:
 *          Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuReST: Robust pupil tracking
 *          for real-time pervasive eye tracking, Symposium on Eye Tracking Research and
 *          Applications (ETRA), 2018, https://doi.org/10.1145/3204493.3204578.
 *
 *          Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuRe: Robust pupil detection
 *          for real-time pervasive eye tracking, Computer Vision and Image Understanding,
 *          2018, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2018.02.002.
 *
 **/

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
    void writeDebugFrame();

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
};

#endif // HCMLAB_PUPILTRACKER_H