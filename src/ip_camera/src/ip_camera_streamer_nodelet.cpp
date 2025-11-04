#include "ip_camera/ip_camera_streamer_nodelet.hpp"

#include <pluginlib/class_list_macros.h>

namespace ip_camera {

void IPCameraStreamerNodelet::onInit() {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle pnh = getPrivateNodeHandle();
    streamer_ = std::make_shared<IPCameraStreamer>(nh, pnh);
}

}

PLUGINLIB_EXPORT_CLASS(ip_camera::IPCameraStreamerNodelet, nodelet::Nodelet)
