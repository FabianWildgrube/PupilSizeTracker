#include <iostream>
#include <string>
#include <sstream>
#include <cstring>

#include <boost/asio.hpp>

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

using boost::asio::ip::tcp;

int main(int argc, char **argv)
{
    try
    {
        boost::asio::io_context io_context;
        tcp::socket serverSocket(io_context);
        serverSocket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 9876));

        std::cout << "Connected to server\n";

        boost::asio::streambuf response_buffer;

        //load video and send metadata
        cv::VideoCapture inputCapture;
        inputCapture.open("/videos/test_short.mp4");
        auto videoWidth = inputCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        auto videoHeight = inputCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        auto fps = inputCapture.get(cv::CAP_PROP_FPS);
        double bytesPerPixel = 3; //only if it's CV_8U3!!

        double * metaData = new double[4];
        metaData[0] = videoWidth;
        metaData[1] = videoHeight;
        metaData[2] = bytesPerPixel;
        metaData[3] = fps;

        boost::asio::write(serverSocket, boost::asio::buffer(reinterpret_cast<const char *>(metaData), 4*sizeof(double)));

        delete[] metaData;

        if (!inputCapture.isOpened()) {
            std::cout << "Could not open video";
            return EXIT_FAILURE;
        }

        std::cout << "opened video";
        cv::Mat camera_frame_raw;
        size_t ts = 0;

        char * matData = new char[static_cast<int>(videoWidth) * static_cast<int>(videoHeight) * 3];

        while (true) {
            inputCapture >> camera_frame_raw;
            if (camera_frame_raw.empty()) {
                break; // End of video.
            }

            for (int i = 0; i < videoHeight; ++i) {
                for (int j = 0; j < videoWidth; ++j) {
                    auto pixel = camera_frame_raw.at<cv::Vec3b>(i,j);
                    for (int k = 0; k < 3; ++k) {
                        *(matData + i * static_cast<int>(videoWidth) * 3 + j * 3 + k) = pixel[k];
                    }
                }
            }

            boost::asio::write(serverSocket, boost::asio::buffer(matData, videoWidth * videoHeight * 3));
            std::cout << "Sent frame " << ts << "\n";

            auto bytes_transferred = boost::asio::read(serverSocket, response_buffer,
                                                       boost::asio::transfer_exactly(4 * sizeof(float)));
            const char* pupilDataTrackingFrameBuffer = boost::asio::buffer_cast<const char*>(response_buffer.data());
            float leftPupilDiameter = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer));
            float leftPupilConfidence = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 1 * sizeof(float)));
            float rightPupilDiameter = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 2 * sizeof(float)));
            float rightPupilConfidence = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 3 * sizeof(float)));

            std::cout << "  Left: " << leftPupilDiameter << " (c: " << leftPupilConfidence << "), Right: " << rightPupilDiameter << " (c: " << rightPupilConfidence << ")\n";

            ts++;
        }

        delete[] matData;
        std::cout << "Done sending video. Exiting. \n";
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return EXIT_SUCCESS;
}