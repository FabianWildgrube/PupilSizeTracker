#include "hcmlabeyeextractor.h"

#include "util/hcmutils.h"

#include <cstdlib>
#include <sstream>
#include <fstream>
#include <memory>
#include <thread>
#include <chrono>
#include <iostream>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

HCMLabEyeExtractor::HCMLabEyeExtractor() : m_currentLandmarksPacketIsEmpty(true), m_currentLandmarksPacketTimestamp(0)
{
}

mediapipe::Status HCMLabEyeExtractor::initIrisTrackingGraph()
{
    hcmutils::logInfo("Initialize the calculator graph.");
    std::string calculator_graph_config_contents;
    auto configLoaded = mediapipe::file::GetContents(m_IrisTrackingGraphConfigFile, &calculator_graph_config_contents);
    if (!configLoaded.ok())
    {
        return configLoaded;
    }

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    auto initialized = m_irisTrackingGraph.Initialize(config);

    if (initialized.ok())
    {
        ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarksPoller, m_irisTrackingGraph.AddOutputStreamPoller(m_kOutputStreamFaceLandmarks));
        m_landmarksPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(landmarksPoller));
    }
    return initialized;
}

mediapipe::Status HCMLabEyeExtractor::init()
{
    MP_RETURN_IF_ERROR(initIrisTrackingGraph());
    MP_RETURN_IF_ERROR(m_irisTrackingGraph.StartRun({}));

    std::thread pollerThread([&] {
        processLandmarkPackets(m_landmarksPoller);
    });
    m_landmarksPollerThread = std::make_unique<std::thread>(std::move(pollerThread));

    return mediapipe::OkStatus();
}

mediapipe::Status HCMLabEyeExtractor::stop()
{
    MP_RETURN_IF_ERROR(m_irisTrackingGraph.CloseInputStream(m_kInputStream));

    //join landmarkspoller thread
    m_landmarksPollerThread->join();

    return m_irisTrackingGraph.WaitUntilDone();
}

IrisDiameters HCMLabEyeExtractor::process(const cv::Mat &inputFrame, size_t framenr, cv::Mat &rightEye, cv::Mat &leftEye)
{
    //push inputFrame into graph
    if (!pushFrameIntoGraph(inputFrame, framenr).ok())
    {
        hcmutils::logError("Could not push frame into graph!");
        return {-1.0f, -1.0f};
    }

    //wait to allow the landmarksPacketPoller to get the data (if there is any)
    int waitIntervalMs = 3;
    int loopCount = 0;
    const int maxWaitLoops = 30;

    while ((m_currentLandmarksPacketIsEmpty || m_currentLandmarksPacketTimestamp < framenr) && loopCount < maxWaitLoops)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(waitIntervalMs));
        loopCount++;
    }

    mediapipe::Packet packetToUse;

    if (loopCount < maxWaitLoops)
    {
        m_currentLandmarksPacketMutex.lock();
        packetToUse = m_currentLandmarksPacket;
        m_lastLandmarksPacket = m_currentLandmarksPacket;
        m_currentLandmarksPacketIsEmpty = true;
        m_currentLandmarksPacketMutex.unlock();
    }
    else
    {
        //use m_lastLandmarksPacket because the current one took way too long
        std::cout << "Graph did not produce a packet for frame " << framenr << " in " << waitIntervalMs * maxWaitLoops << "ms\n";
        packetToUse = m_lastLandmarksPacket;
    }

    // dummy coordinates in case mediapipe can't give us any in time    
    EyesData eyesData = {
        {0.4 * inputFrame.cols, 0.4 * inputFrame.rows, std::max(30.0, 0.01 * inputFrame.cols)},
        {0.6 * inputFrame.cols, 0.6 * inputFrame.rows, std::max(30.0, 0.01 * inputFrame.cols)},
        framenr
    };

    if (!packetToUse.IsEmpty()) {
        eyesData = extractIrisData(packetToUse, inputFrame.cols, inputFrame.rows);
        renderCroppedEyeFrame(inputFrame, eyesData.right, rightEye);
        renderCroppedEyeFrame(inputFrame, eyesData.left, leftEye);
    } else {
        renderCroppedEyeFrame(inputFrame, eyesData.right, rightEye);
        renderCroppedEyeFrame(inputFrame, eyesData.left, leftEye);
    }

    return {eyesData.left.diameter, eyesData.right.diameter};
}

mediapipe::Status HCMLabEyeExtractor::pushFrameIntoGraph(const cv::Mat &inputFrame, size_t timecode)
{
    auto mediapipeFrame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, inputFrame.cols, inputFrame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat mediapipeFrameAsMat = mediapipe::formats::MatView(mediapipeFrame.get());
    inputFrame.copyTo(mediapipeFrameAsMat);

    // Send image packet into the graph.
    MP_RETURN_IF_ERROR(m_irisTrackingGraph.AddPacketToInputStream(m_kInputStream, mediapipe::Adopt(mediapipeFrame.release()).At(mediapipe::Timestamp(timecode))));
    // std::cout << "Frame pushed " << timecode << "\n";

    return mediapipe::OkStatus();
}

void HCMLabEyeExtractor::processLandmarkPackets(const std::unique_ptr<mediapipe::OutputStreamPoller> &poller)
{
    // poll for landmark packets
    while (true)
    {
        mediapipe::Packet packet;
        if (!poller->Next(&packet)) //this call blocks
        {
            //no more landmarks -> exit
            break;
        }
        else
        {
            if (!packet.IsEmpty())
            {
                // std::cout << "packet at " << packet.Timestamp().Value() << "\n";
                if (m_currentLandmarksPacketMutex.try_lock())
                {
                    m_currentLandmarksPacket = packet;
                    m_currentLandmarksPacketIsEmpty = false;
                    m_currentLandmarksPacketTimestamp = packet.Timestamp().Value();
                    m_currentLandmarksPacketMutex.unlock();
                }
                else
                {
                    hcmutils::logError("Landmarkpacket was mutexed, couldn't write the latest one!");
                    // do nothing to prevent deadlock :O
                }
            }
        }
    }
}

/**
 * extracts normalized Iris Landmarks from a landmark packet into the eyesDataFrames vector (as absolute pixel coordinates).
 *
 * The indices used to extract the iris landmarks assume that the 10 iris data points are stored as the last 10 landmarks in the packet.
 * And they should have the following layout:
 *      0: LeftIris_Center,
 *      1: LeftIris_Right,
 *      2: LeftIris_Top,
 *      3: LeftIris_Left,
 *      4: LeftIris_Bottom,
 *
 *      5: RightIris_Center,
 *      6: RightIris_Left,
 *      7: RightIris_Top,
 *      8: RightIris_Right,
 *      9: RightIris_Bottom  <- this is the last landmark in the packet's list
 */
EyesData HCMLabEyeExtractor::extractIrisData(const mediapipe::Packet &landmarksPacket, const int &imageWidth, const int &imageHeight)
{
    auto &output_landmarks = landmarksPacket.Get<mediapipe::NormalizedLandmarkList>();

    auto &right_iris_center_landmark = output_landmarks.landmark(output_landmarks.landmark_size() - 5);
    auto &right_iris_right_landmark = output_landmarks.landmark(output_landmarks.landmark_size() - 2);
    auto &right_iris_left_landmark = output_landmarks.landmark(output_landmarks.landmark_size() - 4);

    auto &left_iris_center_landmark = output_landmarks.landmark(output_landmarks.landmark_size() - 10);
    auto &left_iris_right_landmark = output_landmarks.landmark(output_landmarks.landmark_size() - 9);
    auto &left_iris_left_landmark = output_landmarks.landmark(output_landmarks.landmark_size() - 7);

    IrisData rightIrisData(right_iris_center_landmark.x() * imageWidth,
                           right_iris_center_landmark.y() * imageHeight,
                           hcmutils::GetDistance(right_iris_right_landmark.x() * imageWidth,
                                                 right_iris_right_landmark.y() * imageHeight,
                                                 right_iris_left_landmark.x() * imageWidth,
                                                 right_iris_left_landmark.y() * imageHeight));

    IrisData leftIrisData(left_iris_center_landmark.x() * imageWidth,
                          left_iris_center_landmark.y() * imageHeight,
                          hcmutils::GetDistance(left_iris_right_landmark.x() * imageWidth,
                                                left_iris_right_landmark.y() * imageHeight,
                                                left_iris_left_landmark.x() * imageWidth,
                                                left_iris_left_landmark.y() * imageHeight));

    return {leftIrisData, rightIrisData, landmarksPacket.Timestamp().Value()};
}

bool HCMLabEyeExtractor::renderCroppedEyeFrame(const cv::Mat &camera_frame, const IrisData &irisData, cv::Mat &outputFrame)
{
    auto maxEyeWidth = irisData.diameter * 2.0; //the iris is roughly 1/2 of the total eye size

    auto outputSideLength = maxEyeWidth + 2.0 * m_eyeOutputVideoPadding;
    cv::Size outputSize(outputSideLength, outputSideLength);

    cv::Mat eyeOutputMat(outputSize, camera_frame.type());
    eyeOutputMat = cv::Scalar(255, 0, 0); //fill with red;

    auto topLeftX = std::max(irisData.centerX - (maxEyeWidth / 2.0) - m_eyeOutputVideoPadding, 0.0);
    auto topLeftY = std::max(irisData.centerY - (maxEyeWidth / 2.0) - m_eyeOutputVideoPadding, 0.0);
    auto bottomRightX = std::min(irisData.centerX + (maxEyeWidth / 2.0) + m_eyeOutputVideoPadding, camera_frame.cols * 1.0);
    auto bottomRightY = std::min(irisData.centerY + (maxEyeWidth / 2.0) + m_eyeOutputVideoPadding, camera_frame.rows * 1.0);

    auto eyeCropWindowWidth = bottomRightX - topLeftX;
    auto eyeCropWindowHeight = bottomRightY - topLeftY;

    if (eyeCropWindowWidth > outputSideLength || eyeCropWindowHeight > outputSideLength)
    {
        hcmutils::logError("Eyecropwindow is too big!");
        return false;
    }

    cv::Rect eyeSourceRoi(topLeftX, topLeftY, eyeCropWindowWidth, eyeCropWindowHeight);
    cv::Mat croppedEyeMat = camera_frame(eyeSourceRoi);

    //ensure that the Iris center is always at the center of our output video
    auto irisCenterInSourceRoiX = irisData.centerX - topLeftX;
    auto irisCenterInSourceRoiY = irisData.centerY - topLeftY;
    auto outputTopLeftCoordinateX = outputSideLength / 2.0 - irisCenterInSourceRoiX;
    auto outputTopLeftCoordinateY = outputSideLength / 2.0 - irisCenterInSourceRoiY;

    cv::Rect eyeTargetRoi(outputTopLeftCoordinateX, outputTopLeftCoordinateY, eyeCropWindowWidth, eyeCropWindowHeight);
    cv::Mat eyeOutputTargetMat = eyeOutputMat(eyeTargetRoi);

    croppedEyeMat.copyTo(eyeOutputTargetMat);

    outputFrame.resize(eyeOutputTargetMat.cols * eyeOutputTargetMat.rows);
    eyeOutputTargetMat.copyTo(outputFrame);

    return true;
}