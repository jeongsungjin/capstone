#include "ip_camera/ip_camera_streamer.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "ip_camera_streamer_cpp");

    IPCameraStreamer streamer;
    streamer.run();

    return 0;
}
