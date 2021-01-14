#include <string>

namespace hcmutils
{
    std::string extractFileNameFromPath(std::string path, const std::string &extension);

    void showProgress(const std::string &label, int progress, int max);
    void endProgressDisplay();

    void logInfo(const std::string &message);
    void logError(const std::string &message);

    float GetDistance(float x0, float y0, float x1, float y1);
} // namespace hcmutils
