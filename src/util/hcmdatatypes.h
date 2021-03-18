#ifndef HCMLAB_DATATYPES_H
#define HCMLAB_DATATYPES_H

#include <string>
#include <sstream>

struct RawPupilData
{
    float diameter;
    float confidence;
    long long int ts;
};

struct PupilData
{
    float diameter;
    float diameterRelativeToIris;
    float confidence;
    long long int ts;

    PupilData(RawPupilData rawData, float irisDiameter):
        diameter(rawData.diameter),
        diameterRelativeToIris(rawData.diameter / irisDiameter),
        confidence(rawData.confidence),
        ts(rawData.ts)
    {}
};

struct PupilTrackingDataFrame
{
    PupilData left;
    PupilData right;
};

struct IrisData
{
    float centerX;
    float centerY;
    float diameter;

    IrisData(float cx, float cy, float d) : centerX(cx), centerY(cy), diameter(d) {}

    std::string toJSONString() const
    {
        std::ostringstream jsonStream;
        jsonStream << "{"
                   << "\"centerX\": " << centerX << ","
                   << "\"centerY\": " << centerY << ","
                   << "\"diameter\": " << diameter
                   << "}";

        return jsonStream.str();
    }
};

struct IrisDiameters
{
    float left;
    float right;

    IrisDiameters(float l, float r): left(l), right(r) {};
};

struct EyesData
{
    IrisData left;
    IrisData right;
    size_t frame_nr;

    EyesData(IrisData l, IrisData r, size_t f) : left(l), right(r), frame_nr(f) {}

    std::string toJSONString() const
    {
        std::ostringstream jsonStream;
        jsonStream << "{"
                   << "\"leftEye\": " << left.toJSONString() << ","
                   << "\"rightEye\": " << right.toJSONString() << ","
                   << "\"frame_nr\": " << frame_nr
                   << "}";

        return jsonStream.str();
    }
};

#endif // HCMLAB_DATATYPES_H