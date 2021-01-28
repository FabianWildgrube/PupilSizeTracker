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
#include "outputwriters/hcmlabpupildataoutputwriter.h"
#include "outputwriters/hcmlabpupildatacsvwriter.h"
#include "outputwriters/hcmlabpupildatassiwriter.h"
#include "hcmlabeyeextractor.h"
#include "hcmlabpupildetector.h"

#include "mediapipe/framework/port/commandlineflags.h"

DEFINE_string(input_video_path, "",
              "Full path of video to load. Only '.mp4' files are supported at the moment!");

DEFINE_bool(render_pupil_tracking, false,
            "Whether videos of the eyes with overlayed pupil measurements should be rendered for debugging inspection."
            "False by default");

DEFINE_bool(render_face_tracking, false,
            "Whether video of the face with overlayed face tracking should be rendered for debugging inspection."
            "False by default");

DEFINE_bool(output_as_csv, true,
            "Whether the pupil measurements should be saved in a '.csv' file."
            "true by default");

DEFINE_bool(output_as_ssi, false,
            "Whether the pupil measurements should be saved in a '.stream' file for use with SSI."
            "False by default");

DEFINE_string(output_dir, "./",
              "Directory where the output video files and csv-file with the pupil data should be saved to. "
              "Needs to be supplied with a trailing '/'!"
              "If not provided, the current working directory is used.");

DEFINE_string(output_base_name, "",
              "Base file name of the output files. Will be appended by LEFT_EYE, PUPIL_DATA, etc."
              "If not provided, the name of the input video file is used.");

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_input_video_path == "")
    {
        hcmutils::logError("Please provide a video to work with via the 'input_video_path' command line argument");
        hcmutils::logInfo("Exiting");
        return EXIT_FAILURE;
    }

    // use input file name in case no output file name was provided
    std::string inputFileName = hcmutils::extractFileNameFromPath(FLAGS_input_video_path, ".mp4");

    std::string outputBaseName = FLAGS_output_base_name;
    if (outputBaseName == "")
    {
        outputBaseName = inputFileName;
    }

    std::string outputDirPath = FLAGS_output_dir + inputFileName + "/";
    hcmutils::createDirectoryIfNecessary(outputDirPath);
    // extract eyes from input video
    HCMLabEyeExtractor eyeExtractor(outputDirPath, outputBaseName, FLAGS_render_face_tracking);
    auto extractorResults = eyeExtractor.run(FLAGS_input_video_path);

    if (extractorResults == HCMLabEyeExtractor::EMPTY_OUTPUT)
    {
        hcmutils::logInfo("Exiting");
        return EXIT_FAILURE;
    }

    hcmutils::logInfo("Wrote EyeExtractor results to:\n\t\t\t" + extractorResults.leftEyeVideoFilePath + "\n\t\t\t" + extractorResults.rightEyeVideoFilePath + "\n\t\t\t" + extractorResults.eyeTrackingJsonFilePath + "\n\t\t\tTracking: " + extractorResults.eyeTrackingOverlayVideoFilePath);

    /*
    EyeExtractorOutput extractorResults =
        {"/videos/output/test/test_LEFT-EYE.mp4",
         "/videos/output/test/test_RIGHT-EYE.mp4",
         "",
         "",
         25};
*/

    // detect pupils
    std::vector<PupilData>
        leftEyeData;
    std::vector<PupilData> rightEyeData;

    std::string leftPupilDebugVideoPath = FLAGS_render_pupil_tracking ? outputDirPath + outputBaseName + "_LEFT_PUPIL_TRACK.mp4" : "";
    std::string rightPupilDebugVideoPath = FLAGS_render_pupil_tracking ? outputDirPath + outputBaseName + "_RIGHT_PUPIL_TRACK.mp4" : "";

    auto detectLeft = [&] {
        hcmutils::logInfo("Starting Left pupil detector");
        HCMLabPupilDetector detectorLeft(leftPupilDebugVideoPath, FLAGS_render_pupil_tracking);
        detectorLeft.run(extractorResults.leftEyeVideoFilePath, leftEyeData);
    };

    auto detectRight = [&] {
        hcmutils::logInfo("Starting Right pupil detector");
        HCMLabPupilDetector detectorRight(rightPupilDebugVideoPath, FLAGS_render_pupil_tracking);
        detectorRight.run(extractorResults.rightEyeVideoFilePath, rightEyeData);
    };

    hcmutils::runMultiThreaded({detectLeft, detectRight});

    // write out the data
    std::vector<std::unique_ptr<HCMLabPupilDataOutputWriter_I>> outputWriters;

    if (FLAGS_output_as_csv)
    {
        outputWriters.push_back(std::make_unique<HCMLabPupilDataCSVWriter>(outputDirPath, outputBaseName));
    }

    if (FLAGS_output_as_csv)
    {
        outputWriters.push_back(std::make_unique<HCMLabPupilDataSSIWriter>(outputDirPath, outputBaseName, extractorResults.inputFPS));
    }

    for (const auto &writer : outputWriters)
    {
        writer->write(leftEyeData, rightEyeData);
    }

    //render pupil tracks and face tracking next to each other for debug purposes
    if (FLAGS_render_face_tracking || FLAGS_render_pupil_tracking)
    {
        hcmutils::renderAsCombinedVideo(
            {extractorResults.eyeTrackingOverlayVideoFilePath,
             leftPupilDebugVideoPath,
             rightPupilDebugVideoPath},
            outputDirPath + outputBaseName + "_DEBUG.mp4");
    }

    // clean up the temporary files
    hcmutils::logInfo("Cleaning up");
    hcmutils::removeFileIfPresent(extractorResults.leftEyeVideoFilePath);
    hcmutils::removeFileIfPresent(extractorResults.rightEyeVideoFilePath);
    hcmutils::removeFileIfPresent(extractorResults.eyeTrackingOverlayVideoFilePath);
    hcmutils::removeFileIfPresent(leftPupilDebugVideoPath);
    hcmutils::removeFileIfPresent(rightPupilDebugVideoPath);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    hcmutils::logDuration(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000);
    hcmutils::logInfo("Done");
    return EXIT_SUCCESS;
}