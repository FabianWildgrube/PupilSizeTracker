#ifndef HCMLAB_UTILS_H
#define HCMLAB_UTILS_H
#include <string>
#include <vector>
#include <functional>

namespace hcmutils
{
    bool fileExists(std::string filePath);
    void createDirectoryIfNecessary(const std::string &dirPath);
    void removeFileIfPresent(const std::string &path);
    std::string extractFileNameFromPath(std::string path, const std::string &extension);

    void showProgress(const std::string &label, int progress, int max);
    void endProgressDisplay();

    void logDuration(long long duration);
    void logInfo(const std::string &message);
    void logError(const std::string &message);
    void logProgramEnd();

    float GetDistance(float x0, float y0, float x1, float y1);

    void runMultiThreaded(const std::vector<std::function<void()>> &functions);

    void renderAsCombinedVideo(const std::vector<std::string> &inputVideoPaths, const std::string &outputPath);

} // namespace hcmutils

#endif // HCMLAB_UTILS_H
