echo "###### Building Dummy client ######"
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures=true src:dummyclient
echo "###### Running Dummy Client ######"
GLOG_logtostderr=1 bazel-bin/src/dummyclient
