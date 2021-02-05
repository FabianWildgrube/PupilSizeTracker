/**
 * Pupil-Size-Tracking without head-mounted eye-tracking cameras:
 * Track the size of both of a human's pupils just from the video of their face.
 * 
 * For best results, use a camera that records in the infrared spectrum (750nm and above).
 * Illuminate the eyes with an infrared light source, that preferrably is a point light, in order to keep
 * its reflection in the eyes as small as possible.
 * The picture should have good contrast, too.
 * 
 * This program expects a video file containing a human face as input.
 * The video must contain both eyes as well as at least part of the nose, cheek and forehead.
 * For best results stay as close to the face as possible while honoring these requirements.
 * 
 * The ouput is a .json file containing pixel coordinates for both irises in each frame
 * and a .csv file containing the pupil-diameter and confidence of that value for both eyes in each frame.
 * 
 * If the "render_pupil_tracking" flag is set, two videos of the eye regions with the tracked pupil-diameter
 * and confidence rendered onto them are created as well.
 * 
 * Written by Fabian Wildgrube, HCMLab 2020-2021
 * 
 * Makes use of:
 *  Google's Mediapipe:
 *          Camillo Lugaresi, et al., MediaPipe: A Framework for Building Perception Pipelines,
 *          2019, https://arxiv.org/abs/1906.08172
 *
 *  PuRe and PuReSt:
 *          Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuReST: Robust pupil tracking
 *          for real-time pervasive eye tracking, Symposium on Eye Tracking Research and
 *          Applications (ETRA), 2018, https://doi.org/10.1145/3204493.3204578.
 * 
 *          Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuRe: Robust pupil detection
 *          for real-time pervasive eye tracking, Computer Vision and Image Understanding,
 *          2018, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2018.02.002.
 * 
 **/

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>

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