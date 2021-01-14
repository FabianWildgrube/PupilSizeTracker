# Copyright 2021 Fabian Wildgrube

licenses(["notice"])

package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "hcmlab_utils",
    srcs = [
        "hcmdatatypes.h",
        "hcmutils.h",
        "hcmutils.cc",
    ],
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:opencv_core",
    ],
)

