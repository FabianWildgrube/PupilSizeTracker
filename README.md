# HCMLAB Pupil Size Tracker
C++ Application that detects the pupil size of a human from a video of a face. No eyetracking hardware required. Can output the detected pupil diameter and the confidence of that value to `.csv` and SSI's `.stream` formats.

For best results, use a camera that records in the infrared spectrum (750nm and above). Illuminate the eyes with an infrared light source, preferrably a point light, in order to keep its reflection in the eyes as small as possible.

## Parameters
* `--input_video_path` *[required]*
    
    absolute path of video to load. Only '.mp4' files are supported at the moment!

* `--output_dir` *[defalt: `./`]*

    Directory where the output video files and csv-file with the pupil data should be saved to.

    __Must__ be supplied with a trailing '/'!

* `--output_base_name` *[defalt: name of the input file]*

    Base file name of the output files. Will be appended by LEFT_EYE, PUPIL_DATA, etc.

* `--output_as_csv` *[defalt: `true`]*

    Whether the pupil measurements should be saved in a '.csv' file.

* `--output_as_ssi` *[defalt: `false`]*

    Whether the pupil measurements should be saved in a '.stream' file for use with SSI.

* `--render_pupil_tracking` *[defalt: `false`]*

    Whether videos of the eyes with overlayed pupil measurements should be rendered for debugging inspection.

* `--render_face_tracking` *[defalt: `false`]*

    Whether video of the face with overlayed face tracking should be rendered for debugging inspection.

## Technical usage notes
* The repo contains a `Dockerfile` which sets up a linux container with all the necessary dependencies (mainly Google's `mediapipe`).
* To easily configure the program's parameters, modify the file `buildAndRunHCMLabPupilSizeTracker.sh` and use it to run the program

> If you want to build the project outside that container, you will most likely have to perform all of the steps from the Dockerfile. And fight an epic battle with mediapipe because getting that thing to build properly is like taming a dragon ;)
>
> Building is handled by [Bazel](https://bazel.build/), a language-agnostic build and dependency management tool.

## Setup with Docker

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
        --name hcmlabpupiltracking hcmlabpupiltracking:latest \
    ```
3. Modify `buildAndRunHCMLabPupilSizeTracker.sh` and change `--input_video_path` to the file you want to analyze
3. Run the script to build and execute the pupilTracker
    ```sh
    # in the docker container @ /hcmlabpupiltracking/
    ./buildAndRunHCMLabPupilSizeTracker.sh
    ```

## Running the custom iris tracking script (in the docker container)

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