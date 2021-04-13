echo "###### Building HCMLabPupilMeasurer ######"
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures=true src:hcmlab_run_pupilsizetracking
echo "###### Running HCMLabPupilMeasurer ######"
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/SingleEye.mp4 --output_dir=/videos/output/ --render_debug_video=true --input_is_single_eye=true