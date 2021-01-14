#include "hcmlabpupildatacsvwriter.h"

#include "src/util/hcmutils.h"

#include <fstream>
#include <iostream>

HCMLabPupilDataCSVWriter::HCMLabPupilDataCSVWriter(std::string outputDirPath, std::string baseFileName) : HCMLabPupilDataOutputWriter_I(outputDirPath, baseFileName + "_PUPIL_DATA.csv") {}

void HCMLabPupilDataCSVWriter::write(const std::vector<PupilData> &leftEyeData, const std::vector<PupilData> &rightEyeData)
{
    std::ofstream csvFile(m_outputDirPath + m_outputFileName);
    csvFile << "ts, left_diam, left_conf, right_diam, right_conf\n";

    size_t ctr = 0;
    size_t leftIdx = 0, rightIdx = 0;

    const std::string missingDataPoint = "-1,-1";

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
                csvFile << missingDataPoint << ",";
            }
        }
        else
        {
            csvFile << missingDataPoint << ",";
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
                csvFile << missingDataPoint;
            }
        }
        else
        {
            csvFile << missingDataPoint;
        }

        csvFile << "\n";

        ctr++;
    }
}