echo "###### Building Pupil Tracking Server ######"
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures=true src:hcmlab_run_pupilsizetrackingserver
echo "###### Running Pupil Tracking Server ######"
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetrackingserver
