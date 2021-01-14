#ifndef HCMLAB_PUPILDATACSVWRITER_H
#define HCMLAB_PUPILDATACSVWRITER_H

#include <string>
#include <vector>

#include "hcmlabpupildataoutputwriter.h"
#include "mediapipe/examples/desktop/hcmlab_pupilsizetracking/util/hcmdatatypes.h"

/**
 * Writes out the pupilData into a csv file following the format: 
 * ts, left_diam, left_conf, right_diam, right_conf
 * 
 * empty datapoints are encoded as "-1"
 */
class HCMLabPupilDataCSVWriter : public HCMLabPupilDataOutputWriter_I
{
public:
    HCMLabPupilDataCSVWriter(std::string outputDirPath, std::string baseFileName);
    ~HCMLabPupilDataCSVWriter(){};

    void write(const std::vector<PupilData> &leftEyeData, const std::vector<PupilData> &rightEyeData) override;
};
#endif // HCMLAB_PUPILDATACSVWRITER_H