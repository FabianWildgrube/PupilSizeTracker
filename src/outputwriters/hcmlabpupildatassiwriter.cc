#include "hcmlabpupildatassiwriter.h"

#include <fstream>
#include <algorithm>

HCMLabPupilDataSSIWriter::HCMLabPupilDataSSIWriter(std::string outputDirPath, std::string baseFileName, float inputFPS) : HCMLabPupilDataOutputWriter_I(outputDirPath, baseFileName + "_PUPIL_DATA.stream"), m_fps(inputFPS) {}

void HCMLabPupilDataSSIWriter::write(const std::vector<PupilData> &leftEyeData, const std::vector<PupilData> &rightEyeData)
{
    createSSIHeaderFile(std::max(leftEyeData.size(), rightEyeData.size()));

    std::ofstream streamFile(m_outputDirPath + m_outputFileName + "~");

    size_t ctr = 0;
    size_t leftIdx = 0, rightIdx = 0;

    const float missingDataPoint = -1.0f;

    while (leftIdx < leftEyeData.size() || rightIdx < rightEyeData.size())
    {
        if (leftIdx < leftEyeData.size())
        {
            auto &leftPupil = leftEyeData[leftIdx];
            if (leftPupil.ts == ctr)
            {
                streamFile.write((const char *)&(leftPupil.diameter), sizeof(leftPupil.diameter));
                streamFile.write((const char *)&(leftPupil.confidence), sizeof(leftPupil.confidence));
                leftIdx++;
            }
            else
            {
                streamFile.write((const char *)&(missingDataPoint), sizeof(float));
                streamFile.write((const char *)&(missingDataPoint), sizeof(float));
            }
        }
        else
        {
            streamFile.write((const char *)&(missingDataPoint), sizeof(float));
            streamFile.write((const char *)&(missingDataPoint), sizeof(float));
        }

        if (rightIdx < rightEyeData.size())
        {
            auto &rightPupil = rightEyeData[rightIdx];
            if (rightPupil.ts == ctr)
            {
                streamFile.write((const char *)&(rightPupil.diameter), sizeof(rightPupil.diameter));
                streamFile.write((const char *)&(rightPupil.confidence), sizeof(rightPupil.confidence));
                rightIdx++;
            }
            else
            {
                streamFile.write((const char *)&(missingDataPoint), sizeof(float));
                streamFile.write((const char *)&(missingDataPoint), sizeof(float));
            }
        }
        else
        {
            streamFile.write((const char *)&(missingDataPoint), sizeof(float));
            streamFile.write((const char *)&(missingDataPoint), sizeof(float));
        }
        ctr++;
    }
}

void HCMLabPupilDataSSIWriter::createSSIHeaderFile(const size_t nrOfDataPoints)
{
    std::ofstream streamFile(m_outputDirPath + m_outputFileName);

    streamFile << "<?xml version=\"1.0\" ?>\n"
               << "<stream ssi-v=\"2\">\n"
               << "    <info ftype=\"BINARY\" sr=\"" << m_fps << "\" dim=\"4\" byte=\"" << sizeof(float) << "\" type=\"FLOAT\" />\n"
               << "    <meta />\n"
               << "    <chunk from=\"0.000000\" to=\"" << nrOfDataPoints / m_fps << "\" byte=\"0\" num=\"" << nrOfDataPoints << "\"/>\n"
               << "</stream>\n";
}