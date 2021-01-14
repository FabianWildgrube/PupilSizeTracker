#ifndef HCMLAB_PUPILDATASSIWRITER_H
#define HCMLAB_PUPILDATASSIWRITER_H

#include <string>
#include <vector>

#include "hcmlabpupildataoutputwriter.h"
#include "mediapipe/examples/desktop/hcmlab_pupilsizetracking/util/hcmdatatypes.h"

/**
 * Writes out the pupilData into a binary '.stream~' file, which can be used with SSI
 * A header file '.stream' is created as well describing the data shape of the binary file.
 * 
 * Empty datapoints are represented as -1.0f
 */
class HCMLabPupilDataSSIWriter : public HCMLabPupilDataOutputWriter_I
{
public:
    HCMLabPupilDataSSIWriter(std::string outputDirPath, std::string baseFileName, float inputFPS);
    ~HCMLabPupilDataSSIWriter(){};

    void write(const std::vector<PupilData> &leftEyeData, const std::vector<PupilData> &rightEyeData) override;

private:
    /// generates the '.stream' xml file describing the shape of the data encoded in the '.stream~' file in the way ssi expects it
    void createSSIHeaderFile(const size_t nrOfDataPoints);

    float m_fps; // frames per second of the recorded data
};
#endif // HCMLAB_PUPILDATASSIWRITER_H