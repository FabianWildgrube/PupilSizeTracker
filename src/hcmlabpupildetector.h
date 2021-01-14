#ifndef HCMLAB_PUPILDETECTOR_H
#define HCMLAB_PUPILDETECTOR_H

#include <string>
#include <vector>

#include "pure_pupiltracking/PuRe.h"
#include "pure_pupiltracking/PuReST.h"
#include "pure_pupiltracking/PuReUtils.h"

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

struct PupilData
{
    int diameter;
    float confidence;
    Timestamp ts;
};

class HCMLabPupilDetector
{
public:
    HCMLabPupilDetector();
    HCMLabPupilDetector(bool renderDebugVideo, std::string debugOutputPath);
    ~HCMLabPupilDetector();

    void run(const std::string &inputFilePath, std::vector<PupilData> &trackedPupils); //supports only .mp4 video files a.t.m.!

private:
    void drawPupilOutline(cv::Mat &img_RGB, cv::Point center, double radius);
    void putPupilInfoText(cv::Mat &img_RGB, int diameter, float confidence);

    Pupil m_pupil;
    PuRe m_pure;
    PuReST m_purest;

    cv::VideoCapture m_inputVideoCapture;
    cv::VideoWriter m_debugVideoWriter;
    bool m_renderDebugVideo;
    std::string m_debugOutputPath;
};
#endif // HCMLAB_PUPILDETECTOR_H