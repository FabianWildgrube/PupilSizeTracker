#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>

#include <boost/asio.hpp>

#include "util/hcmutils.h"
#include "util/hcmdatatypes.h"
#include "hcmlabpupiltracker.h"

#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

DEFINE_string(input_video_path,

"",
"Full path of video to load. Only '.mp4' files are supported at the moment!");

DEFINE_bool(render_pupil_tracking,

false,
"Whether videos of the eyes with overlayed pupil measurements should be rendered for debugging inspection."
"False by default");

DEFINE_bool(render_face_tracking,

false,
"Whether video of the face with overlayed face tracking should be rendered for debugging inspection."
"False by default");

DEFINE_bool(output_as_csv,

true,
"Whether the pupil measurements should be saved in a '.csv' file."
"true by default");

DEFINE_bool(output_as_ssi,

false,
"Whether the pupil measurements should be saved in a '.stream' file for use with SSI."
"False by default");

DEFINE_string(output_dir,

"./",
"Directory where the output video files and csv-file with the pupil data should be saved to. "
"Needs to be supplied with a trailing '/'!"
"If not provided, the current working directory is used.");

DEFINE_string(output_base_name,

"",
"Base file name of the output files. Will be appended by LEFT_EYE, PUPIL_DATA, etc."
"If not provided, the name of the input video file is used.");

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_input_video_path == "") {
        hcmutils::logError("Please provide a video to work with via the 'input_video_path' command line argument");
        hcmutils::logInfo("Exiting");
        return EXIT_FAILURE;
    }

    // use input file name in case no output file name was provided
    std::string inputFileName = hcmutils::extractFileNameFromPath(FLAGS_input_video_path, ".mp4");

    std::string outputBaseName = FLAGS_output_base_name;
    if (outputBaseName == "") {
        outputBaseName = inputFileName;
    }

    std::string outputDirPath = FLAGS_output_dir + inputFileName + "/";
    hcmutils::createDirectoryIfNecessary(outputDirPath);


    //load video and run all stuff
    cv::VideoCapture inputCapture;
    inputCapture.open(FLAGS_input_video_path);
    auto videoLength = inputCapture.get(cv::CAP_PROP_FRAME_COUNT);
    auto videoWidth = inputCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    auto videoHeight = inputCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto fps = inputCapture.get(cv::CAP_PROP_FPS);

    if (!inputCapture.isOpened()) {
        hcmutils::logError("Could not open " + FLAGS_input_video_path);
        return EXIT_FAILURE;
    }

    hcmutils::logInfo("Opened video");
    cv::Mat camera_frame_raw;
    size_t ts = 0;

    HCMLabPupilTracker pupilTracker(videoWidth, videoHeight, fps, true,
                                    true, false, outputDirPath, outputBaseName);

    if (!pupilTracker.init()) {
        hcmutils::logError("Could not initialize PupilTracker");
        return EXIT_FAILURE;
    }

    while (true) {
        inputCapture >> camera_frame_raw;
        if (camera_frame_raw.empty()) {
            break; // End of video.
        }

        pupilTracker.process(camera_frame_raw, ts);

        hcmutils::showProgress("Processing", ts, videoLength);
        ts++;
    }
    hcmutils::endProgressDisplay();

    if (!pupilTracker.stop()) {
        hcmutils::logError("Error stopping PupilTracker");
        return EXIT_FAILURE;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    hcmutils::logDuration(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000);
    hcmutils::logInfo("Done");
    return EXIT_SUCCESS;
}