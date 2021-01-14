# Copyright 2020 Fabian Wildgrube

licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "hcmlab_run_pupilsizetracking",
    srcs = [
        "runHCMLabPupilSizeTracking.cc",
        "hcmlabeyeextractor.h",
        "hcmlabeyeextractor.cc",
        "hcmlabpupildetector.h",
        "hcmlabpupildetector.cc"
    ],
    deps = [
        "//src/util:hcmlab_utils",
        "//src/outputwriters:hcmlab_pupildata_outputwriters",
        "//src/pure_pupiltracking:pure_pupil_tracking",
        "@mediapipe//mediapipe/graphs/iris_tracking:iris_tracking_cpu_video_input_deps",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:commandlineflags",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status",
    ],
)