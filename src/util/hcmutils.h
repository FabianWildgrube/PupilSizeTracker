#ifndef HCMLAB_UTILS_H
#define HCMLAB_UTILS_H
#include <string>
#include <vector>
#include <functional>

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

namespace hcmutils
{
    bool fileExists(std::string filePath);
    void createDirectoryIfNecessary(const std::string &dirPath);
    void removeFileIfPresent(const std::string &path);
    std::string extractFileNameFromPath(std::string path, const std::string &extension);

    std::string getCurrentTimeString();

    void showProgress(const std::string &label, int progress, int max);
    void endProgressDisplay();

    void logDuration(long long duration);
    void logInfo(const std::string &message);
    void logError(const std::string &message);
    void logProgramEnd();

    float GetDistance(float x0, float y0, float x1, float y1);

    void runMultiThreaded(const std::vector<std::function<void()>> &functions);

    void writeIntoFrame(cv::Mat& output, const cv::Mat& input, int targetX, int targetY, int maxWidth, int maxHeight);

    void renderAsCombinedVideo(const std::vector<std::string> &inputVideoPaths, const std::string &outputPath);

} // namespace hcmutils

#endif // HCMLAB_UTILS_H
