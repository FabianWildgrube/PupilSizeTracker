#ifndef HCMLAB_PUPILTRACKER_H
#define HCMLAB_PUPILTRACKER_H

#include "util/hcmdatatypes.h"

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

/**
 * Interface for pupil tracking entities that can process a stream of video frames and return tracking data for two pupils
 * Follows the paradigm of calling:
 * I_HCMLabPupilTracker tracker = new ...someImplementation()...;
 * tracker.init();
 * 
 * while(frame)
 *      tracker.process(frame);
 * 
 * tracker.stop(); //important for shutting down any resources the tracker has opened
 *  
*/
class I_HCMLabPupilTracker
{
public:
    virtual bool init() = 0;

    /// Tracks human pupils and their size in the given inputFrame. Meant for online use (i.e. call this function for each frame of a stream of frames).
    /// @param inputFrame - a single frame of the input video containing a human face
    /// @param frameNr - number of the frame within the source video, used as a timecode
    virtual PupilTrackingDataFrame process(const cv::Mat &inputFrame, size_t frameNr) = 0;

    virtual bool stop() = 0;
};

#endif // HCMLAB_PUPILTRACKER_H