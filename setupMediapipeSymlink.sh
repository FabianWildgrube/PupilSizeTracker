if [ -e mediapipe ]
then
    echo "ok"
else
    ln -s /hcmlabpupiltracking/deps/mediapipe-0.8.2/mediapipe/ ./mediapipe
fi