#include <iostream>
#include <string>

#include <boost/asio.hpp>

#include "util/hcmutils.h"
#include "util/hcmdatatypes.h"
#include "hcmlabpupiltracker.h"

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

using boost::asio::ip::tcp;

std::string make_daytime_string()
{
    using namespace std; // For time_t, time and ctime;
    time_t now = time(0);
    return ctime(&now);
}

int main(int argc, char **argv)
{
    try
    {
        boost::asio::io_context io_context;

        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 9876));

        std::cout << "Listening for connections on port 9876" << "\n";

        for(;;)
        {
            std::cout << "Waiting for new connection\n";
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            std::cout << "Connection accepted" << "\n";

            /// setup buffer
            boost::asio::streambuf read_buffer;

            /// metadata transfer (4 floats: width, height, bbp, fps)
            size_t readBytesNr = 4 * sizeof(double);
            auto bytes_transferred = boost::asio::read(socket, read_buffer,
                                                  boost::asio::transfer_exactly(readBytesNr));
            std::cout << "Received " << bytes_transferred << " Bytes of metadata.\n";

            const char* metaDataBuffer = boost::asio::buffer_cast<const char*>(read_buffer.data());

            double videoWidth = static_cast<double>(*reinterpret_cast<const double *>(metaDataBuffer));
            double videoHeight = static_cast<double>(*reinterpret_cast<const double *>(metaDataBuffer + 1 * sizeof(double)));
            double bytesPerPixel = static_cast<double>(*reinterpret_cast<const double *>(metaDataBuffer + 2 * sizeof(double)));
            double fps = static_cast<double>(*reinterpret_cast<const double *>(metaDataBuffer + 3 * sizeof(double)));
            read_buffer.consume(bytes_transferred);

            std::cout << "Width: " << videoWidth << ", Height: " << videoHeight << ", bbp: " << bytesPerPixel << ", fps: " << fps << "\n";

            HCMLabPupilTracker pupilTracker(static_cast<int>(videoWidth), static_cast<int>(videoHeight), fps, true,
                                            true, true, "/videos/output/test/", "testingServer");

            if (!pupilTracker.init()) {
                hcmutils::logError("Could not initialize PupilTracker");
                pupilTracker.stop();
                break;
            }

            int bytesPerFrame = static_cast<int>(videoWidth * videoHeight * bytesPerPixel);
            char* imageBuffer = new char[bytesPerFrame];

            size_t ts = 0;

            for (;;) {
                boost::system::error_code connection_error;
                bytes_transferred = boost::asio::read(socket, read_buffer,
                                                      boost::asio::transfer_exactly(bytesPerFrame),
                                                      connection_error);
                if (connection_error == boost::asio::error::eof) {
                    read_buffer.consume(bytes_transferred);
                    std::cout << "Connection closed by client\n";
                    break;
                }

                const unsigned char* bufPtr = boost::asio::buffer_cast<const unsigned char*>(read_buffer.data());

                cv::Mat videoFrame(videoHeight, videoWidth, CV_8UC3);

                for (int i = 0; i < videoHeight; ++i) {
                    for (int j = 0; j < videoWidth; ++j) {
                        auto& pixel = videoFrame.at<cv::Vec3b>(i,j);
                        for (int k = 0; k < 3; ++k) {
                            unsigned char value = *(bufPtr + i * static_cast<int>(videoWidth) * 3 + j * 3 + k);
                            pixel[k] = value;
                        }
                    }
                }

                PupilTrackingDataFrame trackingData = pupilTracker.process(videoFrame, ts);
                float pupilMeasurements[] = {trackingData.left.diameter, trackingData.left.diameterRelativeToIris, trackingData.left.confidence, trackingData.right.diameter, trackingData.right.diameterRelativeToIris, trackingData.right.confidence};

                boost::asio::write(socket, boost::asio::buffer(reinterpret_cast<const char *>(&pupilMeasurements), 6 * sizeof(float)), connection_error);
                if (connection_error) {
                    std::cout << "Connection closed by client\n";
                    break;
                }

                read_buffer.consume(bytes_transferred);
                ts++;
            }

            delete[] imageBuffer;
            if (!pupilTracker.stop()) {
                std::cout << "Error stopping pupilTracker\n";
            }
        }

    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return EXIT_SUCCESS;

    /**
     * Set up server on port 9876
     *
     * On connection, expect:
     *  frame width, height, fps, renderDebugFlag
     *
     * init pupilTracker
     *
     * if failure: send NEG
     * else: send ACK
     *
     * On receive packet:
     *  accumulate into cv::Mat
     *
     * On full packet:
     *  process cv::Mat in pupilTracker
     *  send EyesDataFrame
     *
     * On connection close:
     *  stop pupilTracker
    */
}