# Copyright 2020 Fabian Wildgrube
package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "pure_pupil_tracking",
    srcs = [
        "PupilDetectionMethod.h", 
        "PupilDetectionMethod.cc",
        "PuRe.h",
        "PuRe.cc",
        "PuReST.h",
        "PuReST.cc",
        "PupilTrackingMethod.h",
        "PupilTrackingMethod.cc",
        "PuReUtils.h",
        "PuReUtils.cc"
    ],
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:opencv_core",
    ],
)

