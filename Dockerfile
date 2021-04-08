# Copyright 2021 Fabian Wildgrube
# based on the mediapipe dockerfile

FROM ubuntu:18.04

WORKDIR /io
WORKDIR /hcmlabpupiltracking

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        nano\
        wget \
        unzip \
        python3-dev \
        python3-opencv \
        python3-pip \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev \
        libopencv-calib3d-dev \
        libopencv-features2d-dev \
        software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && apt-get install -y openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools
RUN pip3 install wheel
RUN pip3 install future
RUN pip3 install six==1.14.0
RUN pip3 install tensorflow==1.14.0
RUN pip3 install tf_slim

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install bazel
ARG BAZEL_VERSION=3.4.1
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

# Download mediapipe
# RUN git clone --branch 0.8.2 https://github.com/google/mediapipe.git /hcmlabpupiltracking/deps/mediapipe && \
COPY ./deps/mediapipe-0.8.2 /hcmlabpupiltracking/deps/mediapipe

# Copy the project into the container
COPY . /hcmlabpupiltracking/

# Build the pupiltracker
RUN bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures=true src:hcmlab_run_pupilsizetracking

# Has to be done manually inside the container otherwise container starts and stops instantly
# CMD ./setupMediapipeSymlink.sh

