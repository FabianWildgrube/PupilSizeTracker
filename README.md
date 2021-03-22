# Pupil Size Tracker - Server
Dockerized TCP-based server that will return left and right pupil size for each video frame it is sent. Mainly meant to be used in conjunction with the pupilTracking plugin for SSI (https://github.com/hcmlab/ssi).

Internally uses the same C++ Application that is described on the `master` branch.

__Limitations:__
* Single client only with synchronous communication
* Listens on the hardcoded port `9876`
* A debug video showing the input image and the tracking is rendered into the directory `/videos` in the server's docker container

## Setup
The repo contains a `Dockerfile` which sets up a linux container with all the necessary dependencies (mainly Google's `mediapipe`).

> If you want to build the project outside that container, you will most likely have to perform all of the steps from the Dockerfile. And fight an epic battle with mediapipe because getting that thing to build properly is like taming a dragon ;)
>
> Building is handled by [Bazel](https://bazel.build/), a language-agnostic build and dependency management tool.

0. Download mediapipe ([release 0.8.2](https://github.com/google/mediapipe/releases/tag/0.8.2)) and copy the files into the directory `deps/mediapipe-0.8.2`

1. Build the Docker Container using the Dockerfile (this will take a rather long time, so go get lunch):
    ```sh
    # in the root of this repository:
    docker build --tag=hcmlabpupiltrackingserver .
    ```

2. Start the docker container the first time with volumes for video and code files.
   
    __IMPORTANT: don't forget to expose port `9876`__
   
    ```sh
    # anywhere on the host (put the command on one line)
    docker run \
        -v <absolute path to a directory for the videofiles>:/videos \
        -v <absolute path to the repos directory>:/hcmlabpupiltracking \
        -it \
        -p 9876:9876 \
        --name hcmlabpupiltrackingserver hcmlabpupiltrackingserver:latest \
    # this will execute the "CMD" of the image, which starts the server
    ```
3. Create a symlink to the mediapipe directory, otherwise mediapipes `tflite` models will not be found during runtime
    ```sh
    # in the docker container @ /hcmlabpupiltracking/
    ln -s /hcmlabpupiltracking/deps/mediapipe-0.8.2/mediapipe/ ./mediapipe
    ```

## Starting the server if it's not running but setup was already done once
1. Start the server's container
    ```sh
    # anywhere on the host
    docker start -i hcmlabpupiltrackingserver
    ```
2. Start the server through the helper script
    ```sh
    # inside the docker container, in the directory /hcmlabpupiltracking
    ./buildAndRunPupilTrackingServer.sh
    ```

## Testing without SSI
Use the `runDummyClient.sh` script to test the server without setting up SSI. It uses the `dummyClient.cc` executable.

1. Make sure the server is running
2. Start a second terminal within the server's container:
    ```sh
    # anywhere on the host
    docker exec -it hcmlabpupiltrackingserver /bin/bash
    ```
3. Execute the helper script
    ```sh
    # inside the docker container, in the directory /hcmlabpupiltracking
    ./runDummyClient.sh  
    ```