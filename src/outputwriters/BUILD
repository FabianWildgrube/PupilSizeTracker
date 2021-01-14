# Copyright 2021 Fabian Wildgrube

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples/desktop/hcmlab_pupilsizetracking:__subpackages__"])

cc_library(
    name = "hcmlab_pupildata_outputwriters",
    srcs = [
        "hcmlabpupildataoutputwriter.h",
        "hcmlabpupildatacsvwriter.h",
        "hcmlabpupildatacsvwriter.cc",
        "hcmlabpupildatassiwriter.h",
        "hcmlabpupildatassiwriter.cc",
    ],
    deps = [
        "//mediapipe/examples/desktop/hcmlab_pupilsizetracking/util:hcmlab_utils",
    ],
)

