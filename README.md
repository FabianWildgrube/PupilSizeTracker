# HCMLAB Pupil Size Tracker
C++ Application that detects the pupil size of a human from a video (complete file or stream of images via TCP) of a face. No eyetracking hardware required. Can output the detected pupil diameter and the confidence of that value to `.csv` and SSI's `.stream` formats.

When starting the program as a TCP server, frames can be sent to the server and pupilSize and confidence for both eyes will be returned as a package of four bytes.

For best results, use a camera that records in the infrared spectrum (750nm and above). Illuminate the eyes with an infrared light source, preferrably a point light, in order to keep its reflection in the eyes as small as possible.

## Technical usage notes
* The repo contains a `Dockerfile` which sets up a linux container with all the necessary dependencies (mainly Google's `mediapipe`).

> If you want to build the project outside that container, you will most likely have to perform all of the steps from the Dockerfile. And fight an epic battle with mediapipe because getting that thing to build properly is like taming a dragon ;)
>
> Building is handled by [Bazel](https://bazel.build/), a language-agnostic build and dependency management tool.

## Setup with Docker

0. Download mediapipe ([release 0.8.2](https://github.com/google/mediapipe/releases/tag/0.8.2)) and copy the files into the directory `deps/mediapipe-0.8.2`

1. Build the Docker Container using the Dockerfile (this will take a rather long time, so go get lunch):
    ```sh
    # in the root of this repository:
    docker build --tag=hcmlabpupiltracking .
    ```

2. Start the docker container the first time with volumes for video and code files. 
    > This way you can edit the code on your machine and easily pass videos into the container and get rendered videos out of the container.
    ```sh
    # anywhere on the host (put the command on one line)
    docker run \
        -v <absolute path to a directory containing your videofiles>:/videos \
        -v <absolute path to the repos directory>:/hcmlabpupiltracking \
        -it \
        -p 9876:9876 \
        --name hcmlabpupiltracking hcmlabpupiltracking:latest \
    # this will execute the "CMD" of the image, which starts the server
    ```

> The following only if you're not using the container as a server
3. Modify `buildAndRunHCMLabPupilSizeTracker.sh` and change `--input_video_path` to the file you want to analyze
4. Run the script to build and execute the pupilTracker
    ```sh
    # in the docker container @ /hcmlabpupiltracking/
    ./buildAndRunHCMLabPupilSizeTracker.sh
    ```

## Running the custom iris tracking script

1. Start an interactive shell in the docker container 
    ```sh
    docker start hcmlabpupiltracking -i
    ```

2. Manually build and run the Pupil Size Tracker
    ```sh
    # in the docker container @ /hcmlabpupiltracking/
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 src:hcmlab_run_pupilsizetracking
    GLOG_logtostderr=1 bazel-bin/src/hcmlab_run_pupilsizetracking \
        --input_video_path=... \
        --output_base_name=... \
        --output_dir=... \
        --output-as-ssi=true \
        --render-face-tracking=true \
        --render-pupil-tracking=true \
        
    ```

    __OR__

    Simply use the script `buildAndRunHCMLabPupilSizeTracker.sh` which [is located here](/buildAndRunHCMLabPupilSizeTracker.sh)
    ```sh
    # in the docker container @ /hcmlabpupiltracking/
    ./buildAndRunHCMLabPupilSizeTracker.sh
    ```


## Parameters (for non-server use, i.e. on existing video files)
* `--input_video_path` *[required]*
    
    absolute path of video to load. Only '.mp4' files are supported at the moment!

* `--output_dir` *[default: `./`]*

    Directory where the output video files and csv-file with the pupil data should be saved to.

    __Must__ be supplied with a trailing '/'!

* `--output_base_name` *[default: name of the input file]*

    Base file name of the output files. Will be appended by LEFT_EYE, PUPIL_DATA, etc.

* `--output_as_csv` *[default: `true`]*

    Whether the pupil measurements should be saved in a '.csv' file.

* `--output_as_ssi` *[default: `false`]*

    Whether the pupil measurements should be saved in a '.stream' file for use with SSI.

* `--render_pupil_tracking` *[default: `false`]*

    Whether videos of the eyes with overlayed pupil measurements should be rendered for debugging inspection.

* `--render_face_tracking` *[default: `false`]*

    Whether video of the face with overlayed face tracking should be rendered for debugging inspection.