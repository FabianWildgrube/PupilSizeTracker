#include "hcmlabpupildatassiwriter.h"

#include <fstream>
#include <algorithm>

HCMLabPupilDataSSIWriter::HCMLabPupilDataSSIWriter(std::string outputDirPath, std::string baseFileName, float inputFPS) : HCMLabPupilDataOutputWriter_I(outputDirPath, baseFileName + "_PUPIL_DATA.stream"), m_fps(inputFPS) {}

void HCMLabPupilDataSSIWriter::write(const std::vector<PupilTrackingDataFrame> &eyeTrackingData)
{
    createSSIHeaderFile(eyeTrackingData.size());

    std::ofstream streamFile(m_outputDirPath + m_outputFileName + "~");

    for (size_t ctr = 0; ctr < eyeTrackingData.size(); ++ctr)
    {
        auto &leftPupil = eyeTrackingData[ctr].left;
        streamFile.write((const char *)&(leftPupil.diameter), sizeof(leftPupil.diameter));
        streamFile.write((const char *)&(leftPupil.confidence), sizeof(leftPupil.confidence));

        auto &rightPupil = eyeTrackingData[ctr].right;
        streamFile.write((const char *)&(rightPupil.diameter), sizeof(rightPupil.diameter));
        streamFile.write((const char *)&(rightPupil.confidence), sizeof(rightPupil.confidence));
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