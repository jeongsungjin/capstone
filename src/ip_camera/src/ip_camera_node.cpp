// ROS2 multi-camera manager: creates batch_size IPCameraStreamer nodes and spins them

#include <rclcpp/rclcpp.hpp>

#include "ip_camera/ip_camera_streamer.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto options = rclcpp::NodeOptions{};
    // enable intra-process comms for zero-copy when running in the same container
    options.use_intra_process_comms(true);
    auto node = std::make_shared<IPCameraStreamer>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
