#ifndef HCMLAB_PUPILDATAOUTPUTWRITER_I_H
#define HCMLAB_PUPILDATAOUTPUTWRITER_I_H

#include <string>
#include <vector>
#include <iostream>

#include "mediapipe/examples/desktop/hcmlab_pupilsizetracking/util/hcmdatatypes.h"

/**
 * Interface for writing out the pupil measurement data to files.
 * Create a subclass which implements the 'write' method for each filetype that should be supported for export.
 */
class HCMLabPupilDataOutputWriter_I
{
public:
    HCMLabPupilDataOutputWriter_I() : m_outputDirPath("."), m_outputFileName("output"){};
    HCMLabPupilDataOutputWriter_I(std::string outputDirPath, std::string baseFileName) : m_outputDirPath(outputDirPath), m_outputFileName(baseFileName){};
    ~HCMLabPupilDataOutputWriter_I(){};

    virtual void write(const std::vector<PupilData> &leftEyeData, const std::vector<PupilData> &rightEyeData) = 0;

protected:
    std::string m_outputDirPath;
    std::string m_outputFileName;
};
#endif // HCMLAB_PUPILDATAOUTPUTWRITER_I_H