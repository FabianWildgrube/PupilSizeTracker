#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <chrono>

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

        std::cout << "opened video\n";
        cv::Mat camera_frame_raw;
        size_t ts = 0;

        const int bytesPerFrame = static_cast<int>(videoWidth) * static_cast<int>(videoHeight) * 3;
        char * matData = new char[bytesPerFrame];

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        while (true) {
            
	std::chrono::steady_clock::time_point beginMeasure = std::chrono::steady_clock::now();

            inputCapture >> camera_frame_raw;
            if (camera_frame_raw.empty()) {
                break; // End of video.
            }

            unsigned char * rawFrameData = camera_frame_raw.data;
            memcpy(matData, rawFrameData, bytesPerFrame);
    std::chrono::steady_clock::time_point endMeasure = std::chrono::steady_clock::now();
	std::cout << "Copying in: " << std::chrono::duration_cast<std::chrono::microseconds>(endMeasure - beginMeasure).count() / 1000.f << " ms. ";


	beginMeasure = std::chrono::steady_clock::now();

            boost::asio::write(serverSocket, boost::asio::buffer(matData, bytesPerFrame));
            //std::cout << "Sent frame " << ts << "\n";

    endMeasure = std::chrono::steady_clock::now();
	std::cout << "Sending: " << std::chrono::duration_cast<std::chrono::microseconds>(endMeasure - beginMeasure).count() / 1000.f << " ms. ";


	beginMeasure = std::chrono::steady_clock::now();

            auto bytes_transferred = boost::asio::read(serverSocket, response_buffer,
                                                       boost::asio::transfer_exactly(6 * sizeof(float)));

	endMeasure = std::chrono::steady_clock::now();
	std::cout << "Receiving: " << std::chrono::duration_cast<std::chrono::microseconds>(endMeasure - beginMeasure).count() / 1000.f << " ms.\n";

            const char* pupilDataTrackingFrameBuffer = boost::asio::buffer_cast<const char*>(response_buffer.data());
            float leftPupilDiameter = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer));
            float leftPupilDiameterRelative = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 1 * sizeof(float)));
            float leftPupilConfidence = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 2 * sizeof(float)));
            float rightPupilDiameter = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 3 * sizeof(float)));
            float rightPupilDiameterRelative = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 4 * sizeof(float)));
            float rightPupilConfidence = static_cast<float>(*reinterpret_cast<const float *>(pupilDataTrackingFrameBuffer + 5 * sizeof(float)));

            std::cout << "  Left: " << leftPupilDiameter << ", relative: " << leftPupilDiameterRelative << " (c: " << leftPupilConfidence << "), Right: " << rightPupilDiameter << ", relative: " << rightPupilDiameterRelative << " (c: " << rightPupilConfidence << ")\n";
            response_buffer.consume(bytes_transferred);

            ts++;
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000;
        std::cout << ts << " frames processed in " << duration << " seconds " << " => Speed: " << ts / duration << " fps.\n";

        delete[] matData;
        std::cout << "Done sending video. Exiting. \n";
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return EXIT_SUCCESS;
}