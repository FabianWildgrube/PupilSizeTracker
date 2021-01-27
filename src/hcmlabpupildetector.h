#ifndef HCMLAB_PUPILDETECTOR_H
#define HCMLAB_PUPILDETECTOR_H

#include <string>
#include <vector>
#include <sstream>

#include "pure_pupiltracking/PuRe.h"
#include "pure_pupiltracking/PuReST.h"
#include "pure_pupiltracking/PuReUtils.h"

#include "util/hcmdatatypes.h"

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

class HCMLabPupilDetector
{
public:
    HCMLabPupilDetector();
    HCMLabPupilDetector(std::string debugOutputPath, bool renderDebugVideo);
    ~HCMLabPupilDetector();

    void run(const std::string &inputFilePath, std::vector<PupilData> &trackedPupils); //supports only .mp4 video files a.t.m.!

private:
    void drawPupilOutline(cv::Mat &img_RGB, cv::Point center, double radius);
    void putPupilInfoText(cv::Mat &img_RGB, int diameter, float confidence);
    void putText(cv::Mat &img_RGB, std::string message, const cv::Point &location);

    void optimizeImage(const cv::Mat &img_in_BGR, cv::Mat &img_out_GRAY);
    void adjustImageContrast(cv::Mat &inputImageGRAY, const int &contrast);

    int detectAveragePupilBrightness(const cv::Mat &img_in_GRAY);
    int detectAverageIrisBrightness(cv::Mat &img_in_GRAY);

    void enhanceBrightness(cv::Mat &input_GRAY);
    void enhanceContrast(cv::Mat &input_GRAY);

    Pupil m_pupil;
    PuRe m_pure;
    PuReST m_purest;

    cv::VideoCapture m_inputVideoCapture;
    cv::VideoWriter m_debugVideoWriter;
    bool m_renderDebugVideo;
    std::string m_debugOutputPath;

    std::ostringstream m_debugStringStr;

    bool m_optimizeImage;

    int m_pupilInspectionKernelSize = 30;
    int m_lastFrameContrast = 0;
};
#endif // HCMLAB_PUPILDETECTOR_H