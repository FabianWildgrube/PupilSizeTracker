#include "hcmutils.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <thread>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

namespace hcmutils
{

    bool fileExists(std::string filePath)
    {
        std::ifstream ifile;
        ifile.open(filePath);
        if (ifile.is_open())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool createDirectoryIfNecessary(std::string dirPath)
    {
        mkdir(dirPath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }

    void removeFileIfPresent(const std::string &path)
    {
        if (path != "")
        {
            logInfo("Removing: " + path);
            std::remove(path.c_str());
        }
    }

    std::string extractFileNameFromPath(std::string path, const std::string &extension)
    {
        //remove extension
        path.replace(path.find(extension), extension.length(), "");

        //remove path prefix
        auto last_slash_pos = path.find_last_of("/");
        if (last_slash_pos != std::string::npos)
        {
            return path.substr(last_slash_pos + 1);
        }
        else
        {
            return path;
        }
    }

    void showProgress(const std::string &label, int current, int max)
    {
        int barWidth = 70;

        float progress = static_cast<float>(current) / static_cast<float>(max);

        std::cout << label << ": [";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }

    void endProgressDisplay()
    {
        std::cout << std::endl;
    }

    void log(const std::string &message)
    {
        std::time_t date = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm = *std::localtime(&date);
        std::cout << std::put_time(&tm, "%T") << " " << message << "\n";
    }

    void logInfo(const std::string &message)
    {
        log("---INFO---- " + message);
    }

    void logError(const std::string &message)
    {
        log("---ERROR--- " + message);
    }

    float GetDistance(float x0, float y0, float x1, float y1)
    {
        return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    }

    void runMultiThreaded(const std::vector<std::function<void()>> &functions)
    {
        std::vector<std::thread> threads;

        for (auto &function : functions)
        {
            threads.emplace_back([&] { function(); });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }
    }

    void renderAsCombinedVideo(const std::vector<std::string> &inputVideoPaths, const std::string &outputPath)
    {
        std::vector<cv::VideoCapture> inputVideos;

        const int maxOutputWidth = 1200;
        const int spacing = 20; //spacing between the videos;

        int combinedInputWidth = 0;
        int maxInputHeight = 0;

        int fps = 0;
        int frameCount = 0;

        // open all inputs
        for (auto &inputPath : inputVideoPaths)
        {
            if (inputPath == "")
                continue;

            cv::VideoCapture inputCapture;
            inputCapture.open(inputPath);
            if (inputCapture.isOpened())
            {
                int inputfps = inputCapture.get(cv::CAP_PROP_FPS);
                int inputlength = inputCapture.get(cv::CAP_PROP_FRAME_COUNT);

                if (fps == 0)
                {
                    fps = inputfps;
                }
                if (frameCount == 0)
                {
                    frameCount = inputlength;
                }

                if (fps == inputfps && frameCount == inputlength)
                {
                    inputVideos.push_back(inputCapture);
                    combinedInputWidth += inputCapture.get(cv::CAP_PROP_FRAME_WIDTH) + spacing;
                    maxInputHeight = std::max(maxInputHeight, static_cast<int>(inputCapture.get(cv::CAP_PROP_FRAME_HEIGHT)));
                }
                else
                {
                    //ignore videos that don't have matching fps and frameCount
                }
            }
        }

        combinedInputWidth -= spacing; //because the loop adds one too many

        cv::Size outputSize(std::min(combinedInputWidth, maxOutputWidth), maxInputHeight * std::min(1, maxOutputWidth / combinedInputWidth));

        cv::VideoWriter outputWriter;
        outputWriter.open(outputPath,
                          mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                          fps, outputSize, true);
        if (!outputWriter.isOpened())
        {
            hcmutils::logError("Could not open debugWriter " + outputPath);
            return;
        }

        cv::Mat outputMat = cv::Mat::zeros(outputSize, CV_8UC3);
        cv::Mat assemblyMat = cv::Mat::zeros(maxInputHeight, combinedInputWidth, CV_8UC3);

        // loop over the videos
        for (size_t i = 0; i < frameCount; i++)
        {
            assemblyMat = cv::Scalar(0, 0, 0);
            int currentX = 0;

            for (auto &inputVideo : inputVideos)
            {
                cv::Mat frame;
                inputVideo >> frame;
                if (frame.empty())
                {
                    inputVideo.release();
                    break;
                }

                //center video vertically
                cv::Rect targetRoi(currentX,
                                   std::max((maxInputHeight / 2) - (frame.rows / 2), 0),
                                   frame.cols,
                                   frame.rows);
                cv::Mat targetMat = assemblyMat(targetRoi);
                frame.copyTo(targetMat);

                currentX += frame.cols + spacing;
            }

            cv::resize(assemblyMat, outputMat, outputMat.size(), 0, 0, cv::INTER_AREA);
            outputWriter.write(outputMat);
            showProgress("Rendering debug video", i, frameCount);
        }
        endProgressDisplay();
        outputWriter.release();
    }
} // namespace hcmutils
