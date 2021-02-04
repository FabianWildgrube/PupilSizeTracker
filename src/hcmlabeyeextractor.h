#ifndef HCMLAB_EYEEXTRACTOR_H
#define HCMLAB_EYEEXTRACTOR_H

#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <thread>
#include <mutex>

#include "util/hcmdatatypes.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

/**
 * Extracts Eye position data and crops the eyes from a video of a human face.
 *
 * Renders the eyes into separate video files and writes the eye tracking data into a json file.
 * Paths to these 3 files are returned from the run() method.
 * 
 * Use a single instance of this class to process multiple videos one after another, by calling the run() method with different video files.
 * This saves the setup cost for the mediapipe graph that does the heavy face-tracking lifting.
 */
class HCMLabEyeExtractor
{
public:
    HCMLabEyeExtractor();
    ~HCMLabEyeExtractor(){};

    mediapipe::Status init();

    mediapipe::Status stop();

    /// Extract the eyes from the given inputFrame. Meant for online use (i.e. call this function for each frame of a stream of frames).
    /// @param inputFrame - a single frame of the input video
    /// @param framenr - number of the frame within the source video, used as a timecode
    /// @param rightEye - output parameter. will contain the rightEye after this method returns
    /// @param leftEye - output parameter. will contain the leftEye after this method returns
    void process(const cv::Mat &inputFrame, size_t framenr, cv::Mat &rightEye, cv::Mat &leftEye);

private:
    mediapipe::Status initIrisTrackingGraph();
    mediapipe::Status pushFrameIntoGraph(const cv::Mat &inputFrame, size_t timecode);
    void processLandmarkPackets(const std::unique_ptr<mediapipe::OutputStreamPoller> &poller);
    EyesData extractIrisData(const int &imageWidth, const int &imageHeight);
    bool renderCroppedEyeFrame(const cv::Mat &camera_frame, const IrisData &irisData, cv::Mat &outputFrame);

    std::string m_IrisTrackingGraphConfigFile = "/hcmlabpupiltracking/deps/mediapipe-0.8.2/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt";
    std::string m_kInputStream = "input_video";
    std::string m_kOutputStreamFaceLandmarks = "face_landmarks_with_iris";

    std::unique_ptr<mediapipe::OutputStreamPoller> m_landmarksPoller;
    std::unique_ptr<std::thread> m_landmarksPollerThread;
    mediapipe::Packet m_lastLandmarksPacket;
    std::mutex m_lastLandmarksPacketMutex;

    int m_eyeOutputVideoPadding = 40;

    mediapipe::CalculatorGraph m_irisTrackingGraph;
};
#endif // HCMLAB_EYEEXTRACTOR_H