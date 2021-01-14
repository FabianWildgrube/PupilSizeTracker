# Copyright 2021 Fabian Wildgrube

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples/desktop/hcmlab_pupilsizetracking:__subpackages__"])

cc_library(
    name = "hcmlab_utils",
    srcs = [
        "hcmdatatypes.h",
        "hcmutils.h",
        "hcmutils.cc",
    ],
    deps = [
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:opencv_core",
    ],
)

