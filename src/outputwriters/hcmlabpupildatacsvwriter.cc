#include "hcmlabpupildatacsvwriter.h"

#include "src/util/hcmutils.h"

#include <fstream>
#include <iostream>

HCMLabPupilDataCSVWriter::HCMLabPupilDataCSVWriter(std::string outputDirPath, std::string baseFileName) : HCMLabPupilDataOutputWriter_I(outputDirPath, baseFileName + "_PUPIL_DATA.csv") {}

void HCMLabPupilDataCSVWriter::write(const std::vector<PupilTrackingDataFrame> &eyeTrackingData)
{
    std::ofstream csvFile(m_outputDirPath + m_outputFileName);
    csvFile << "ts, left_diam_abs, left_diam_rel, left_conf, right_diam_abs, right_diam_rel, right_conf\n";

    for (size_t ctr = 0; ctr < eyeTrackingData.size(); ++ctr) {
        csvFile << ctr << ",";

        auto &leftPupil = eyeTrackingData[ctr].left;
        csvFile << leftPupil.diameter << "," << leftPupil.diameterRelativeToIris << ", " << leftPupil.confidence << ",";

        auto &rightPupil = eyeTrackingData[ctr].right;
        csvFile << rightPupil.diameter << "," << rightPupil.diameterRelativeToIris << ", " << rightPupil.confidence;
        csvFile << "\n";

        ctr++;
    }
}