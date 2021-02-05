#include "hcmlabpupildetector.h"

#include "util/hcmutils.h"

#include <sstream>

HCMLabPupilDetector::HCMLabPupilDetector()
    : m_optimizeImage(true),
      m_currentTimestamp(0)
{
}

HCMLabPupilDetector::~HCMLabPupilDetector()
{
}

PupilData HCMLabPupilDetector::process(const cv::Mat &inputFrame)
{
    cv::cvtColor(inputFrame, m_camera_frame_GRAY, cv::COLOR_BGR2GRAY);

    if (m_optimizeImage)
    {
        optimizeImage(inputFrame, m_camera_frame_GRAY);
    }

    cv::Rect roi(0, 0, m_camera_frame_GRAY.cols, m_camera_frame_GRAY.rows);
    m_purest.track(m_currentTimestamp, m_camera_frame_GRAY, roi, m_pupil, m_pure);

    m_currentTimestamp++;

    return {static_cast<float>(m_pupil.diameter()), m_pupil.confidence, m_currentTimestamp};
}

PupilData HCMLabPupilDetector::process(const cv::Mat &inputFrame, cv::Mat &debugOutputFrame)
{
    auto trackingData = process(inputFrame);

    //debug drawing
    cv::cvtColor(m_camera_frame_GRAY, debugOutputFrame, cv::COLOR_GRAY2RGB);
    drawPupilOutline(debugOutputFrame, m_pupil.center, m_pupil.diameter() / 2.0);
    putPupilInfoText(debugOutputFrame, m_pupil.diameter(), m_pupil.confidence);
    putText(debugOutputFrame, m_debugStringStr.str(), cv::Point(5, debugOutputFrame.rows - 25));
    m_debugStringStr.str("");
    m_debugStringStr.clear();

    return trackingData;
}

void HCMLabPupilDetector::optimizeImage(const cv::Mat &img_in_BGR, cv::Mat &img_out_GRAY)
{
    cv::Mat input_GRAY;
    cv::cvtColor(img_in_BGR, input_GRAY, cv::COLOR_BGR2GRAY);

    enhanceBrightness(input_GRAY);
    enhanceContrast(input_GRAY);

    input_GRAY.copyTo(img_out_GRAY);
}

///increases the brightness of a grayscale image inversely to mean and stddev
///(i.e. the lower the mean and stddev, the more the brightness is increased)
void HCMLabPupilDetector::enhanceBrightness(cv::Mat &input_GRAY)
{
    cv::Mat mean_mat;
    cv::Mat stddev_mat;
    cv::meanStdDev(input_GRAY, mean_mat, stddev_mat);
    int mean = static_cast<int>(std::round(mean_mat.at<double>(0)));
    int stddev = static_cast<int>(std::round(stddev_mat.at<double>(0)));

    m_debugStringStr << "M: " << mean << ", S: " << stddev;

    float brightnessFactor;
    auto brightnessIndicator = mean + stddev;
    if (brightnessIndicator < 30)
    {
        brightnessFactor = 4.5f;
    }
    else if (mean < 40)
    {
        brightnessFactor = 3.5f;
    }
    else if (mean < 50)
    {
        brightnessFactor = 3.2f;
    }
    else if (mean < 60)
    {
        brightnessFactor = 2.9f;
    }
    else if (mean < 70)
    {
        brightnessFactor = 2.6f;
    }
    else if (mean < 80)
    {
        brightnessFactor = 2.3f;
    }
    else if (mean < 90)
    {
        brightnessFactor = 2.1f;
    }
    else if (mean < 100)
    {
        brightnessFactor = 1.9f;
    }
    else if (mean < 110)
    {
        brightnessFactor = 1.5f;
    }
    else
    {
        brightnessFactor = 1.0f;
    }

    m_debugStringStr << ", bf: " << brightnessFactor;
    input_GRAY *= brightnessFactor;
}

///enhances the contrast of a grayscale image depending on how much contrast
///already exists between pupil and iris.
void HCMLabPupilDetector::enhanceContrast(cv::Mat &input_GRAY)
{
    auto pupilBrightness = detectAveragePupilBrightness(input_GRAY);
    auto irisBrightness = detectAverageIrisBrightness(input_GRAY);
    auto innerEyeContrast = irisBrightness - pupilBrightness;

    bool adjustContrast = true;
    int contrast = 0;
    if (innerEyeContrast <= 0)
    {
        //iris is darker than pupil -> pupil probably was not at the center
        //no reliable information about inner eyecontrast is possible
        //use the last one available to us
        contrast = m_lastFrameContrast;
    }
    else if (innerEyeContrast < 3)
    {
        contrast = 25;
    }
    else if (innerEyeContrast < 6)
    {
        contrast = 20;
    }
    else if (innerEyeContrast < 9)
    {
        contrast = 15;
    }
    else if (innerEyeContrast < 12)
    {
        contrast = 10;
    }
    else if (innerEyeContrast < 15)
    {
        contrast = 5;
    }
    else if (innerEyeContrast < 18)
    {
        contrast = 2;
    }
    else
    {
        adjustContrast = false;
    }

    m_debugStringStr << ", lc: " << m_lastFrameContrast;

    if (adjustContrast)
    {
        m_debugStringStr << ", IEC: " << innerEyeContrast << ", c: " << contrast;
        adjustImageContrast(input_GRAY, contrast);
        m_lastFrameContrast = contrast;
    }
}

/// modifies the contrast of an image
/// -127 < contrast < 127 is expected!
void HCMLabPupilDetector::adjustImageContrast(cv::Mat &inputImageGRAY, const int &contrast)
{
    double f = (131.0 * (static_cast<double>(contrast) + 127.0)) / (127.0 * (131.0 - static_cast<double>(contrast)));
    auto alpha_c = f;
    auto gamma_c = 127.0 * (1.0 - f);
    cv::addWeighted(inputImageGRAY, alpha_c, inputImageGRAY, 0, gamma_c, inputImageGRAY);
}

/// determines the pupil brightness by averaging over a 20x20 square in the center of the input image.
/// This works because the HCMLabEyeExtractor crops the eyes in a way, that the pupil is in the center of the image most of the time
int HCMLabPupilDetector::detectAveragePupilBrightness(const cv::Mat &img_in_GRAY)
{
    int centerRoiX = (img_in_GRAY.cols - m_pupilInspectionKernelSize) / 2;
    int centerRoiY = (img_in_GRAY.rows - m_pupilInspectionKernelSize) / 2;
    cv::Rect pupilRoi(centerRoiX, centerRoiY, m_pupilInspectionKernelSize, m_pupilInspectionKernelSize);
    cv::Mat pupilMat = img_in_GRAY(pupilRoi);
    const auto pupilBrightness = cv::mean(pupilMat);
    return pupilBrightness[0];
}

/// determines the average brightness of the iris, without the pupil
/// determines this by averaging over 8 20x20 squares distributed as neighbors around a 20x20 square at the center of the image
/// This works because the HCMLabEyeExtractor crops the eyes in a way, that the pupil is in the center of the image most of the time
int HCMLabPupilDetector::detectAverageIrisBrightness(cv::Mat &img_in_GRAY)
{
    //collect 8 kernels into one row and then average over that row of kernels
    int kernelsize = 20;
    cv::Mat aufsammelMat = cv::Mat::zeros(m_pupilInspectionKernelSize, 5 * m_pupilInspectionKernelSize, CV_8UC1);

    int centerX = img_in_GRAY.cols / 2;
    int centerY = img_in_GRAY.rows / 2;

    int aufsammelPos = 0;
    for (int row = 0; row < 2; row++)
    {
        for (int col = -1; col < 2; col++)
        {
            if (col == 0 && row == 0)
                continue; //ignore center square because that is the pupil

            int roiX = centerX + col * m_pupilInspectionKernelSize - m_pupilInspectionKernelSize / 2;
            int roiY = centerY + row * m_pupilInspectionKernelSize - m_pupilInspectionKernelSize / 2;
            cv::Rect kernelRoi(roiX, roiY, m_pupilInspectionKernelSize, m_pupilInspectionKernelSize);
            cv::Mat roiMat = img_in_GRAY(kernelRoi);

            // cv::rectangle(img_in_GRAY, kernelRoi, cv::Scalar(255));

            cv::Rect aufsammelRoi(aufsammelPos, 0, m_pupilInspectionKernelSize, m_pupilInspectionKernelSize);
            cv::Mat aufsammelTargetMat = aufsammelMat(aufsammelRoi);

            roiMat.copyTo(aufsammelTargetMat);

            aufsammelPos += kernelsize;
        }
    }

    const auto irisBrightness = cv::mean(aufsammelMat);
    return irisBrightness[0];
}

void HCMLabPupilDetector::drawPupilOutline(cv::Mat &img_RGB, cv::Point center, double radius)
{
    //the pupil's center
    cv::circle(img_RGB, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    //the outline
    cv::circle(img_RGB, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
}

void HCMLabPupilDetector::putText(cv::Mat &img_RGB, std::string message, const cv::Point &location)
{
    cv::putText(img_RGB, message, location, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
}

void HCMLabPupilDetector::putPupilInfoText(cv::Mat &img_RGB, int diameter, float confidence)
{
    std::ostringstream pupilStrStream;
    pupilStrStream << "diam: " << diameter << ", conf: " << confidence;
    putText(img_RGB, pupilStrStream.str(), cv::Point(5, 20));
}