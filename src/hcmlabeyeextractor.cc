#include "hcmlabeyeextractor.h"

#include "hcmutils.h"

#include <cstdlib>
#include <thread>
#include <fstream>

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

HCMLabEyeExtractor::HCMLabEyeExtractor(std::string tempOutputPath, std::string outputBaseFileName) : m_outputDirPath(tempOutputPath), m_outputBaseFileName(outputBaseFileName), m_inputVideoFps(0), m_inputVideoLength(0)
{
    if (!initIrisTrackingGraph())
    {
        hcmutils::logError("PROLBEM INITIALIZING");
    }
}

bool HCMLabEyeExtractor::initIrisTrackingGraph()
{
    hcmutils::logInfo("Initialize the calculator graph.");
    std::string calculator_graph_config_contents;
    auto configLoaded = mediapipe::file::GetContents(m_IrisTrackingGraphConfigFile, &calculator_graph_config_contents);
    if (!configLoaded.ok())
        return false;

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    auto initialized = m_irisTrackingGraph.Initialize(config);
    return initialized.ok();
}

bool HCMLabEyeExtractor::loadInputVideo(const std::string &inputFilePath)
{
    hcmutils::logInfo("Load the video.");
    m_inputVideoCapture.open(inputFilePath);
    m_inputVideoFps = m_inputVideoCapture.get(cv::CAP_PROP_FPS);
    m_inputVideoLength = m_inputVideoCapture.get(cv::CAP_PROP_FRAME_COUNT);
    return m_inputVideoCapture.isOpened();
}

void HCMLabEyeExtractor::reset()
{
    m_inputVideoCapture.release();
    m_inputVideoFps = 0;
    m_eyesDataFrames.clear();
}

EyeExtractorOutput HCMLabEyeExtractor::run(const std::string &inputFilePath)
{
    reset();

    if (!loadInputVideo(inputFilePath))
    {
        return {"", "", ""};
    }

    if (!runIrisTrackingGraph().ok())
    {
        hcmutils::logError("Mediapipe graph did not return status 'ok'!");
        return {"", "", ""};
    }

    if (m_eyesDataFrames.size() > 0)
    {
        std::string eyeTrackJSONFilePath = m_outputDirPath + m_outputBaseFileName + "_IRIS-DATA.json";
        writeEyesDataToJSONFile(inputFilePath, eyeTrackJSONFilePath);

        //loop over input video again to render cropped eye regions into separate video files
        if (!loadInputVideo(inputFilePath))
        {
            return {"", "", ""};
        }

        std::string leftEyeVideoPath = m_outputDirPath + m_outputBaseFileName + "_LEFT-EYE.mp4";
        std::string rightEyeVideoPath = m_outputDirPath + m_outputBaseFileName + "_RIGHT-EYE.mp4";

        if (!writeCroppedEyesIntoVideoFiles(leftEyeVideoPath, rightEyeVideoPath))
        {
            return {"", "", ""};
        }

        return {leftEyeVideoPath, rightEyeVideoPath, eyeTrackJSONFilePath};
    }
    else
    {
        hcmutils::logError("Mediapipe could not track any face/eye features!");
        return {"", "", ""};
    }
}

mediapipe::Status HCMLabEyeExtractor::runIrisTrackingGraph()
{
    hcmutils::logInfo("Start running the calculator graph.");
    //TODO: add option to render facetracking landmarks for debug purposes
    //ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller outputImagepoller, m_irisTrackingGraph.AddOutputStreamPoller(m_kOutputStreamVideo));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarksPoller, m_irisTrackingGraph.AddOutputStreamPoller(m_kOutputStreamFaceLandmarks));

    MP_RETURN_IF_ERROR(m_irisTrackingGraph.StartRun({}));

    hcmutils::logInfo("Start grabbing and processing frames.");

    // split into two threads: one for pumping the input video frames into the graph and polling the frames with landmarks overlays out of the graph;
    // the second thread polls the graph for landmark packets. Since landmarks are not necessarily detected in every input frame this thread blocks from time to time
    // => if both of those tasks were on the main thread the thread would block and not input the next frame into the graph -> it'll block forever.
    std::vector<std::thread> threads;

    std::atomic<bool> grab_frames(true);

    threads.emplace_back([&] {
        //push the video frames into the graph and render the overlays out
        size_t frame_count = 1;
        cv::Mat camera_frame_raw;
        cv::Mat camera_frame;
        while (grab_frames)
        {
            // Capture opencv camera or video frame.
            m_inputVideoCapture >> camera_frame_raw;
            if (camera_frame_raw.empty())
            {
                grab_frames = false;
                break; // End of video.
            }
            cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

            // Wrap Mat into an ImageFrame.
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            camera_frame.copyTo(input_frame_mat);

            // Send image packet into the graph.
            MP_RETURN_IF_ERROR(m_irisTrackingGraph.AddPacketToInputStream(m_kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_count))));

            frame_count++;
            hcmutils::showProgress("Tracking eyes in video", frame_count, m_inputVideoLength);
        }
        hcmutils::endProgressDisplay();

        MP_RETURN_IF_ERROR(m_irisTrackingGraph.CloseInputStream(m_kInputStream));
    });

    threads.emplace_back([&] {
        // poll for landmark packets
        while (grab_frames)
        {
            mediapipe::Packet landmarksPacket;
            if (!landmarksPoller.Next(&landmarksPacket) && !grab_frames)
            {
                hcmutils::logInfo("No new landmarksPacket");
                break;
            }
            extractIrisData(landmarksPacket, m_inputVideoCapture.get(cv::CAP_PROP_FRAME_WIDTH), m_inputVideoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        }

        std::ostringstream str;
        str << "Extracted landmarks from " << m_eyesDataFrames.size() << " out of " << m_inputVideoLength << " frames!";
        hcmutils::logInfo(str.str());
    });

    for (auto &thread : threads)
    {
        thread.join();
    }

    return m_irisTrackingGraph.WaitUntilDone();
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
void HCMLabEyeExtractor::extractIrisData(const mediapipe::Packet &landmarksPacket, const int &imageWidth, const int &imageHeight)
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

    m_eyesDataFrames.emplace_back(leftIrisData, rightIrisData, landmarksPacket.Timestamp().Value());
}

bool HCMLabEyeExtractor::writeCroppedEyesIntoVideoFiles(const std::string &leftEyeVideoPath, const std::string &rightEyeVideoPath)
{
    //get max diameter to determine the maximum image size for each eye video
    float leftMaxDiameter = 0.0f;
    float rightMaxDiameter = 0.0f;
    for (const EyesData &eyeData : m_eyesDataFrames)
    {
        if (eyeData.left.diameter > leftMaxDiameter)
            leftMaxDiameter = eyeData.left.diameter;
        if (eyeData.right.diameter > rightMaxDiameter)
            rightMaxDiameter = eyeData.right.diameter;
    }

    size_t frame_count = 1;
    size_t currentEyeDataIdx = 0;
    auto lastEyeDataIdx = m_eyesDataFrames.size() - 1;
    auto currentEyeData = m_eyesDataFrames.at(currentEyeDataIdx);
    bool grab_frames = true;

    cv::VideoWriter leftEyeWriter;
    cv::VideoWriter rightEyeWriter;

    while (grab_frames)
    {
        cv::Mat camera_frame_raw;
        m_inputVideoCapture >> camera_frame_raw;
        if (camera_frame_raw.empty())
        {
            grab_frames = false;
            break; // End of video.
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        //Some frames of the source video may not have yielded proper tracking data
        //To mitigate this, use the most current tracking data, until we encounter the frame for which we have the next valid tracking data
        if (currentEyeData.frame_nr < frame_count && currentEyeDataIdx < lastEyeDataIdx)
        {
            auto nextEyeData = m_eyesDataFrames.at(currentEyeDataIdx + 1);
            if (nextEyeData.frame_nr == frame_count)
            {
                currentEyeData = nextEyeData;
                currentEyeDataIdx++;
            }
        }

        auto leftOk = renderCroppedEyeFrame(camera_frame, currentEyeData.left, leftMaxDiameter, leftEyeWriter, leftEyeVideoPath);
        auto rightOk = renderCroppedEyeFrame(camera_frame, currentEyeData.right, rightMaxDiameter, rightEyeWriter, rightEyeVideoPath);

        if (!leftOk || !rightOk)
        {
            hcmutils::logError("Error during rendering cropped eye frames");
            return false;
        }

        frame_count++;
        hcmutils::showProgress("Rendering cropped video of left and right eyes", frame_count, m_inputVideoLength);
    }
    hcmutils::endProgressDisplay();

    return true;
}

bool HCMLabEyeExtractor::renderCroppedEyeFrame(
    const cv::Mat &camera_frame,
    const IrisData &irisData,
    float maxIrisDiameter,
    cv::VideoWriter &writer,
    const cv::String &outputVideoPath)
{
    auto maxEyeWidth = maxIrisDiameter * 2.0; //the iris is roughly 1/2 of the total eye size

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

    //initializing writer here, because outputSize is determined within this function
    if (!writer.isOpened())
    {
        writer.open(outputVideoPath,
                    mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                    m_inputVideoFps, outputSize);
        if (!writer.isOpened())
        {
            hcmutils::logError("Could not open writer for " + outputVideoPath);
            return false;
        }
    }

    cv::cvtColor(eyeOutputMat, eyeOutputMat, cv::COLOR_RGB2BGR);
    writer.write(eyeOutputMat);
    return true;
}

void HCMLabEyeExtractor::writeEyesDataToJSONFile(const std::string &inputVideoFileName, const std::string &outputFileName)
{

    hcmutils::logInfo("Writing eye-tracking data to json file " + outputFileName);

    std::ofstream jsonFile(outputFileName);
    jsonFile << "{"
             << "\"associatedVideoFile\": \"" << inputVideoFileName << "\","
             << "\"irisTrackingData\": [";
    size_t count = 0;
    for (const EyesData &eyeData : m_eyesDataFrames)
    {
        jsonFile << eyeData.toJSONString() << (count != m_eyesDataFrames.size() - 1 ? "," : "");
        count++;
    }
    jsonFile << "]"
             << "}";
}