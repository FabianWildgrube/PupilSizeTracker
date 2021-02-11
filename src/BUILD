# Copyright 2020 Fabian Wildgrube

licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "hcmlab_run_pupilsizetrackingcli",
    srcs = [
        "pupilTrackerCLI.cc",
        "hcmlabeyeextractor.h",
        "hcmlabeyeextractor.cc",
        "hcmlabpupildetector.h",
        "hcmlabpupildetector.cc",
        "hcmlabpupiltracker.h",
        "hcmlabpupiltracker.cc"
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

cc_binary(
    name = "hcmlab_run_pupilsizetrackingserver",
    srcs = [
        "pupilTrackerTCPServer.cc",
        "hcmlabeyeextractor.h",
        "hcmlabeyeextractor.cc",
        "hcmlabpupildetector.h",
        "hcmlabpupildetector.cc",
        "hcmlabpupiltracker.h",
        "hcmlabpupiltracker.cc"
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
        "@boost//:asio"
    ],
)

cc_binary(
    name = "dummyclient",
    srcs = [
        "dummyClient.cc"
    ],
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@boost//:asio"
    ],
)