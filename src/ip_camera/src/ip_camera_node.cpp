// ROS2 multi-camera manager: creates batch_size IPCameraStreamer nodes and spins them

#include <rclcpp/rclcpp.hpp>

#include "ip_camera/ip_camera_streamer.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<IPCameraStreamer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
