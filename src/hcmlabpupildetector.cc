#include "hcmlabpupildetector.h"

#include "util/hcmutils.h"

#include <sstream>

HCMLabPupilDetector::HCMLabPupilDetector()
    : m_debugOutputPath(""),
      m_renderDebugVideo(false),
      m_optimizeImage(true)
{
}

HCMLabPupilDetector::HCMLabPupilDetector(std::string debugOutputPath, bool renderDebugVideo)
    : m_debugOutputPath(debugOutputPath),
      m_renderDebugVideo(renderDebugVideo),
      m_optimizeImage(true)
{
}

HCMLabPupilDetector::~HCMLabPupilDetector()
{
    if (m_debugVideoWriter.isOpened())
    {
        m_debugVideoWriter.release();
    }
}

void HCMLabPupilDetector::run(const std::string &inputFilePath, std::vector<PupilData> &trackedPupils)
{
    m_inputVideoCapture.open(inputFilePath);
    if (!m_inputVideoCapture.isOpened())
    {
        hcmutils::logError("Could not open input video: " + inputFilePath);
        return;
    }

    bool grab_frames = true;
    Timestamp ts = 0;
    cv::Mat camera_frame_raw;
    cv::Mat camera_frame_GRAY;

    while (grab_frames)
    {
        // Capture opencv camera or video frame.
        m_inputVideoCapture >> camera_frame_raw;
        if (camera_frame_raw.empty())
        {
            grab_frames = false;
            break; // End of video.
        }
        cv::cvtColor(camera_frame_raw, camera_frame_GRAY, cv::COLOR_BGR2GRAY);

        if (m_optimizeImage)
        {
            optimizeImage(camera_frame_raw, camera_frame_GRAY);
        }

        cv::Rect roi(0, 0, camera_frame_GRAY.cols, camera_frame_GRAY.rows);
        m_purest.track(ts, camera_frame_GRAY, roi, m_pupil, m_pure);
        trackedPupils.push_back({m_pupil.diameter(), m_pupil.confidence, ts});

        if (m_renderDebugVideo)
        {
            cv::Mat outputMatRGB;
            cv::cvtColor(camera_frame_GRAY, outputMatRGB, cv::COLOR_GRAY2RGB);

            drawPupilOutline(outputMatRGB, m_pupil.center, m_pupil.diameter() / 2.0);
            putPupilInfoText(outputMatRGB, m_pupil.diameter(), m_pupil.confidence);

            if (!m_debugVideoWriter.isOpened())
            {
                m_debugVideoWriter.open(m_debugOutputPath,
                                        mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                                        m_inputVideoCapture.get(cv::CAP_PROP_FPS), outputMatRGB.size(), true);
                if (!m_debugVideoWriter.isOpened())
                {
                    hcmutils::logError("Could not open debugWriter " + m_debugOutputPath);
                    m_renderDebugVideo = false;
                }
            }
            cv::cvtColor(outputMatRGB, outputMatRGB, cv::COLOR_RGB2BGR);
            m_debugVideoWriter.write(outputMatRGB);
        }

        ts++;
    }
    m_debugVideoWriter.release();
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

void HCMLabPupilDetector::optimizeImage(const cv::Mat &img_in_BGR, cv::Mat &img_out_GRAY)
{
    cv::Mat input_GRAY;
    cv::cvtColor(img_in_BGR, input_GRAY, cv::COLOR_BGR2GRAY);

    //Determine overall brightness -> make brighter if too low
    auto imageBrightness = cv::mean(input_GRAY)[0];
    if (imageBrightness < 200)
    {
        input_GRAY *= 4;
    }

    auto pupilBrightness = detectAveragePupilBrightness(input_GRAY);

    /* TODO: better determine image quality using stddev to estimate contrast
    cv::Mat mean_mat;
    cv::Mat stddev_mat;
    double mean, stddev;
    cv::meanStdDev(input_GRAY_alt, mean_mat, stddev_mat);
    mean = mean_mat.at<double>(0);
    stddev = stddev_mat.at<double>(0);
    */

    if (pupilBrightness < 30)
    {
        //do nothing, we should be good
    }
    else if (pupilBrightness < 60)
    {
        int contrast = 10;

        adjustImageContrast(input_GRAY, contrast);
    }

    input_GRAY.copyTo(img_out_GRAY);
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
    int width = 20, height = 20;
    int centerRoiX = (img_in_GRAY.cols - width) / 2;
    int centerRoiY = (img_in_GRAY.rows - height) / 2;
    cv::Rect pupilRoi(centerRoiX, centerRoiY, width, height);
    cv::Mat pupilMat = img_in_GRAY(pupilRoi);
    const auto pupilBrightness = cv::mean(pupilMat);
    return pupilBrightness[0];
}