#ifndef HCMLAB_DATATYPES_H
#define HCMLAB_DATATYPES_H

#include <string>
#include <sstream>

struct PupilData
{
    int diameter;
    float confidence;
    long long int ts;
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