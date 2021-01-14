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

#include "hcmutils.h"
#include "hcmlabeyeextractor.h"
#include "hcmlabpupildetector.h"

#include "mediapipe/framework/port/commandlineflags.h"

DEFINE_string(input_video_path, "",
              "Full path of video to load. ");

DEFINE_bool(render_pupil_tracking, false,
            "Whether the pupil tracking data should be rendered onto videos of the eyes for debugging inspection."
            "False by default");

DEFINE_string(output_dir, ".",
              "Directory where the output video files and csv-file with the pupil data should be saved to. "
              "If not provided, the current working directory is used.");

DEFINE_string(output_base_name, "",
              "Base file name of the output files. Will be appended by LEFT_EYE, PUPIL_DATA, etc."
              "If not provided, the name of the input video file is used.");

void writePupilsIntoCSVFile(const std::vector<PupilData> &leftEyeData, const std::vector<PupilData> &rightEyeData, const std::string &outputDirPath, const std::string &baseFileName)
{
    auto fileName = baseFileName + "_PUPIL_DATA.csv";

    std::ofstream csvFile(outputDirPath + fileName);
    csvFile << "ts, left_diam, left_conf, right_diam, right_conf\n";

    size_t ctr = 0;
    size_t leftIdx = 0, rightIdx = 0;

    while (leftIdx < leftEyeData.size() || rightIdx < rightEyeData.size())
    {
        csvFile << ctr << ",";

        if (leftIdx < leftEyeData.size())
        {
            auto &leftPupil = leftEyeData[leftIdx];
            if (leftPupil.ts == ctr)
            {
                csvFile << leftPupil.diameter << "," << leftPupil.confidence << ",";
                leftIdx++;
            }
            else
            {
                LOG(INFO) << "no data for left at " << ctr;
                csvFile << "-1,-1,";
            }
        }
        else
        {
            csvFile << "---,---,";
        }

        if (rightIdx < rightEyeData.size())
        {
            auto &rightPupil = rightEyeData[rightIdx];
            if (rightPupil.ts == ctr)
            {
                csvFile << rightPupil.diameter << "," << rightPupil.confidence;
                rightIdx++;
            }
            else
            {
                LOG(INFO) << "no data for right at " << ctr;
                csvFile << "-1,-1,";
            }
        }
        else
        {
            csvFile << "---,---";
        }

        csvFile << "\n";

        ctr++;
    }
}

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_input_video_path == "")
    {
        hcmutils::logError("Please provide a video to work with via the 'input_video_path' command line argument");
        hcmutils::logInfo("Exiting");
        return EXIT_FAILURE;
    }

    std::string outputBaseName = FLAGS_output_base_name;
    if (outputBaseName == "")
    {
        outputBaseName = hcmutils::extractFileNameFromPath(FLAGS_input_video_path, ".mp4");
    }

    HCMLabEyeExtractor eyeExtractor(FLAGS_output_dir, outputBaseName);
    auto extractorResults = eyeExtractor.run(FLAGS_input_video_path);
    hcmutils::logInfo("Wrote EyeExtractor results to: " + extractorResults.leftEyeVideoFilePath + ", " + extractorResults.rightEyeVideoFilePath + ", " + extractorResults.eyeTrackingJsonFilePath);

    std::vector<PupilData> leftEyeData;
    std::vector<PupilData> rightEyeData;

    std::vector<std::thread> threads;

    threads.emplace_back([&] {
        hcmutils::logInfo("Starting Left eye detector");
        HCMLabPupilDetector detectorLeft(FLAGS_render_pupil_tracking, FLAGS_render_pupil_tracking ? FLAGS_output_dir + outputBaseName + "_LEFT_PUPIL_TRACK.mp4" : "");
        detectorLeft.run(extractorResults.leftEyeVideoFilePath, leftEyeData);
    });

    threads.emplace_back([&] {
        hcmutils::logInfo("Starting Right eye detector");
        HCMLabPupilDetector detectorRight(FLAGS_render_pupil_tracking, FLAGS_render_pupil_tracking ? FLAGS_output_dir + outputBaseName + "_RIGHT_PUPIL_TRACK.mp4" : "");
        detectorRight.run(extractorResults.rightEyeVideoFilePath, rightEyeData);
    });

    for (auto &thread : threads)
    {
        thread.join();
    }

    hcmutils::logInfo("Writing pupil data to csv file");
    writePupilsIntoCSVFile(leftEyeData, rightEyeData, FLAGS_output_dir, outputBaseName);

    // clean up the temporary files
    hcmutils::logInfo("Removing temp eye crop videos");
    std::remove(extractorResults.leftEyeVideoFilePath.c_str());
    std::remove(extractorResults.rightEyeVideoFilePath.c_str());

    hcmutils::logInfo("Done");
    return EXIT_SUCCESS;
}