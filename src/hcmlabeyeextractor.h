#ifndef HCMLAB_EYEEXTRACTOR_H
#define HCMLAB_EYEEXTRACTOR_H

#include <string>
#include <vector>
#include <sstream>
#include <atomic>

#include "util/hcmdatatypes.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

struct EyeExtractorOutput
{
    std::string leftEyeVideoFilePath;
    std::string rightEyeVideoFilePath;
    std::string eyeTrackingOverlayVideoFilePath;
    std::string eyeTrackingJsonFilePath;
    float inputFPS;

    bool operator==(const EyeExtractorOutput &b)
    {
        return (leftEyeVideoFilePath == b.leftEyeVideoFilePath) && (rightEyeVideoFilePath == b.rightEyeVideoFilePath) && (eyeTrackingOverlayVideoFilePath == b.eyeTrackingOverlayVideoFilePath) && (eyeTrackingJsonFilePath == b.eyeTrackingJsonFilePath) && (inputFPS == b.inputFPS);
    }
};

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
    HCMLabEyeExtractor(std::string tempOutputPath, std::string outputBaseFileName, bool renderTrackingOverlays);
    ~HCMLabEyeExtractor(){};

    /// extract the eyes from the passed video. Not threadsafe for multiple videos
    /// @param inputFilePath - absolute path to the input video (must contain a face with two eyes)
    EyeExtractorOutput run(const std::string &inputFilePath); //supports only .mp4 video files a.t.m.!

    static EyeExtractorOutput EMPTY_OUTPUT;

private:
    bool initIrisTrackingGraph();
    bool loadInputVideo(const std::string &inputFilePath);
    void writeEyesDataToJSONFile(const std::string &inputVideoFileNames);
    void extractIrisData(const mediapipe::Packet &landmarksPacket, const int &imageWidth, const int &imageHeight);
    bool writeCroppedEyesIntoVideoFiles();
    bool renderCroppedEyeFrame(const cv::Mat &camera_frame, const IrisData &irisData, float maxIrisDiameter, cv::VideoWriter &writer, const cv::String &outputVideoPath);
    mediapipe::Status runIrisTrackingGraph();

    mediapipe::Status pushInputVideoIntoGraph();
    mediapipe::Status pushInputVideoIntoGraphAndRenderTracking(mediapipe::OutputStreamPoller &outputImagepoller);
    void processLandmarkPackets(mediapipe::OutputStreamPoller &landmarksPoller);
    void renderTrackingOverlays(mediapipe::OutputStreamPoller &outputImagepoller);

    void reset();

    std::string m_IrisTrackingGraphConfigFile = "mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt";
    std::string m_kInputStream = "input_video";
    std::string m_kOutputStreamTrackingOverlays = "output_video";
    std::string m_kOutputStreamFaceLandmarks = "face_landmarks_with_iris";

    int m_eyeOutputVideoPadding = 40;

    std::string m_leftEyeVideoPath;
    std::string m_rightEyeVideoPath;
    std::string m_trackingOverlayVideoPath;
    std::string m_eyeTrackJSONFilePath;

    mediapipe::CalculatorGraph m_irisTrackingGraph;

    cv::VideoCapture m_inputVideoCapture;
    double m_inputVideoFps;
    size_t m_inputVideoLength;

    bool m_renderTrackingOverlaysForDebugging;
    cv::VideoWriter m_trackingOverlaysWriter;

    std::vector<EyesData> m_eyesDataFrames;
};
#endif // HCMLAB_EYEEXTRACTOR_H