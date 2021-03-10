echo "###### Building HCMLabPupilMeasurer ######"
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures=true src:hcmlab_run_pupilsizetracking
echo "###### Running HCMLabPupilMeasurer ######"
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test01/Test01.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test01/Test01_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test02/Test02.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test02/Test02_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test03/Test03.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test03/Test03_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test04/Test04.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test04/Test04_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test05/Test05.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test05/Test05_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test06/Test06.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test06/Test06_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test07/Test07.mp4 --output_dir=/videos/output/ --render_debug_video=true
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking --input_video_path=/videos/IR_Tests_2021-03-10/Test07/Test07_glasses.mp4 --output_dir=/videos/output/ --render_debug_video=true
