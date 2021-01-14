#include "hcmlabpupildetector.h"

#include "hcmutils.h"

#include <sstream>

HCMLabPupilDetector::HCMLabPupilDetector() : m_debugOutputPath(""), m_renderDebugVideo(false) {}

HCMLabPupilDetector::HCMLabPupilDetector(bool renderDebugVideo, std::string debugOutputPath) : m_renderDebugVideo(renderDebugVideo), m_debugOutputPath(debugOutputPath)
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

        cv::Rect roi(0, 0, camera_frame_GRAY.cols, camera_frame_GRAY.rows);
        m_purest.track(ts, camera_frame_GRAY, roi, m_pupil, m_pure);
        trackedPupils.push_back({m_pupil.diameter(), m_pupil.confidence, ts});

        if (m_renderDebugVideo)
        {
            cv::Mat outputMatRGB;
            cv::cvtColor(camera_frame_raw, outputMatRGB, cv::COLOR_BGR2RGB);
            cv::cvtColor(outputMatRGB, outputMatRGB, cv::COLOR_RGB2GRAY);
            cv::cvtColor(outputMatRGB, outputMatRGB, cv::COLOR_GRAY2RGB);

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
}

void HCMLabPupilDetector::drawPupilOutline(cv::Mat &img_RGB, cv::Point center, double radius)
{
    //the pupil's center
    cv::circle(img_RGB, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    //the outline
    cv::circle(img_RGB, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
}

void HCMLabPupilDetector::putPupilInfoText(cv::Mat &img_RGB, int diameter, float confidence)
{
    std::ostringstream pupilStrStream;
    pupilStrStream << "diam: " << diameter << ", conf: " << confidence;
    cv::putText(img_RGB, pupilStrStream.str(), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}