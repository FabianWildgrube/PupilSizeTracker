# HCMLAB Pupil Tracker
Dockerized C++ Application that detects pupil size of a human from a video of a face.

## Setup with Docker
Build the Docker Container using the Dockerfile:
```
# in the root of this repository:
docker build --tag=hcmlabpupiltracking .
```

Start the docker container the first time with volumes for video and code files. This way you can edit the code on your machine and easily pass videos into the container and get rendered videos out of the container.
```
docker run -v …TestVideos:/videos -v <absolute path to ./src>:/hcmlabpupiltracking/src -it --name hcmlabpupiltracking hcmlabpupiltracking:latest
```

## Running the custom iris tracking script (in the docker container)
Start an interactive shell in the docker container (via `docker start hcmlabpupiltracking -i`)
Building custom iris tracking (in the container):
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 src:hcmlab_run_pupilsizetracking
```
> Building is handled by [Bazel](https://bazel.build/), a language-agnostic build and dependency management tool. It is already installed in the docker container, so don't worry about it too much.

Running custom iris tracking (in the container):
```
GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking \
    --render-pupil-tracking=true \
    --output_dir=... \
    --input_video_path=...
```

Or simply use the script `buildAndRunHCMLabPupilMeasurer.sh` which [is located here](/buildAndRunHCMLabPupilMeasurer.sh), in the root of this repo.