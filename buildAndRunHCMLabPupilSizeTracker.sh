echo "###### Building Pupil Tracker CLI ######"
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures=true src:hcmlab_run_pupilsizetrackingcli
echo "###### Running Pupil Tracker CLI ######"
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetrackingcli \
    --input_video_path=/videos/test.mp4 \
    --output_dir=/videos/output/ \
    --render-pupil-tracking=true \
    --render-face-tracking=false \
    --output_as_ssi=true \
    --output_as_csv=true
