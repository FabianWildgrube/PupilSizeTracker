::Alter these variables to your liking to customize the docker container setup
SET imageTag=testserverimg
SET containerName=testserver
SET videoMountDirectory=C:\Users\admin\codingprojects\HCMLab\TestMount

@ECHO OFF
SETLOCAL ENABLEEXTENSIONS
SET parent=%~dp0

Call :DownloadAndExtractMediapipe

ECHO Building Image
docker build --tag=%imageTag% .

ECHO Creating and running container
docker run -v %parent%:/hcmlabpupiltracking -v %videoMountDirectory%:/videos -it -p 9876:9876 --name %containerName% %imageTag%:latest

PAUSE
EXIT /B 0

::Download mediapipe if not already present
:DownloadAndExtractMediapipe
IF NOT EXIST "deps" (
    ECHO Creating deps directory
    mkdir deps
)
cd deps

IF NOT EXIST "mediapipe-0.8.2" (
    IF NOT EXIST "0.8.2.zip" (
        ECHO Downloading mediapipe
        wget https://github.com/google/mediapipe/archive/refs/tags/0.8.2.zip
    )

    ECHO Unzipping mediapipe
    REM unzip "%parent%\deps\0.8.2.zip" to "%parent%\deps\mediapipe-0.8.2\"
    powershell.exe -nologo -noprofile -command "& { $shell = New-Object -COM Shell.Application; $target = $shell.NameSpace('%parent%deps\'); $zip = $shell.NameSpace('%parent%deps\0.8.2.zip'); $target.CopyHere($zip.Items(), 16); }"
    ECHO Unzipped mediapipe to %parent%\deps\mediapipe-0.8.2\
    DEL 0.8.2.zip
)

cd %parent%
EXIT /B 0
