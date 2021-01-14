#include "hcmutils.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace hcmutils
{
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
} // namespace hcmutils
