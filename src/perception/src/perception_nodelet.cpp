#include "perception/perception_nodelet.hpp"

#include <pluginlib/class_list_macros.h>
#include <ros/package.h>

namespace perception_ns {

void PerceptionNodelet::onInit() {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle pnh = getPrivateNodeHandle();

    std::string pkg_path = ros::package::getPath("perception");

    int batch_size;
    pnh.param<int>("batch_size", batch_size, 2);

    NODELET_INFO_STREAM("PerceptionNodelet params: pkg_path=" << pkg_path
                        << ", batch_size=" << batch_size);

    node_ = std::make_unique<PerceptionNode>(pkg_path, batch_size);
    // Nodelet uses shared callback queues, no spin() call here.
}

} // namespace perception_ns

PLUGINLIB_EXPORT_CLASS(perception_ns::PerceptionNodelet, nodelet::Nodelet)
