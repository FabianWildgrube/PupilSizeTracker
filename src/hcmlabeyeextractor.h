#ifndef HCMLAB_EYEEXTRACTOR_H
#define HCMLAB_EYEEXTRACTOR_H

#include <string>
#include <vector>
#include <sstream>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

struct EyeExtractorOutput
{
    std::string leftEyeVideoFilePath;
    std::string rightEyeVideoFilePath;
    std::string eyeTrackingJsonFilePath;
};

struct IrisData
{
    float centerX;
    float centerY;
    float diameter;

    IrisData(float cx, float cy, float d) : centerX(cx), centerY(cy), diameter(d) {}

    std::string toJSONString() const
    {
        std::ostringstream jsonStream;
        jsonStream << "{"
                   << "\"centerX\": " << centerX << ","
                   << "\"centerY\": " << centerY << ","
                   << "\"diameter\": " << diameter
                   << "}";

        return jsonStream.str();
    }
};

struct EyesData
{
    IrisData left;
    IrisData right;
    int64 frame_nr;

    EyesData(IrisData l, IrisData r, int64 f) : left(l), right(r), frame_nr(f) {}

    std::string toJSONString() const
    {
        std::ostringstream jsonStream;
        jsonStream << "{"
                   << "\"leftEye\": " << left.toJSONString() << ","
                   << "\"rightEye\": " << right.toJSONString() << ","
                   << "\"frame_nr\": " << frame_nr
                   << "}";

        return jsonStream.str();
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
    HCMLabEyeExtractor(std::string tempOutputPath, std::string outputBaseFileName);
    ~HCMLabEyeExtractor(){};

    /// extract the eyes from the passed video. Not threadsafe for multiple videos
    /// @param inputFilePath - absolute path to the input video (must contain a face with two eyes)
    EyeExtractorOutput run(const std::string &inputFilePath); //supports only .mp4 video files a.t.m.!

private:
    bool initIrisTrackingGraph();
    bool loadInputVideo(const std::string &inputFilePath);
    void writeEyesDataToJSONFile(const std::string &inputVideoFileName, const std::string &baseFileName);
    void extractIrisData(const mediapipe::Packet &landmarksPacket, const int &imageWidth, const int &imageHeight);
    bool writeCroppedEyesIntoVideoFiles(const std::string &leftEyeVideoPath, const std::string &rightEyeVideoPath);
    bool renderCroppedEyeFrame(const cv::Mat &camera_frame, const IrisData &irisData, float maxIrisDiameter, cv::VideoWriter &writer, const cv::String &outputVideoPath);
    mediapipe::Status runIrisTrackingGraph();

    void reset();

    std::string m_IrisTrackingGraphConfigFile = "iris_tracking_cpu.pbtxt";
    std::string m_kInputStream = "input_video";
    std::string m_kOutputStreamVideo = "output_video";
    std::string m_kOutputStreamFaceLandmarks = "face_landmarks_with_iris";

    int m_eyeOutputVideoPadding = 40;

    std::string m_outputDirPath;
    std::string m_outputBaseFileName;

    mediapipe::CalculatorGraph m_irisTrackingGraph;

    cv::VideoCapture m_inputVideoCapture;
    double m_inputVideoFps;
    size_t m_inputVideoLength;

    std::vector<EyesData> m_eyesDataFrames;
};
#endif // HCMLAB_EYEEXTRACTOR_H