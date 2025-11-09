#include <ros/ros.h>
#include <ros/package.h>

#include "perception/perception_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "perception_node");
    
    std::string pkg_path = ros::package::getPath("perception");

    int batch_size;
    ros::param::param<int>(
        "~batch_size",
        batch_size,
        1
    );

    ROS_INFO_STREAM("ros param list");
    ROS_INFO_STREAM("batch size : " << batch_size);    

    PerceptionNode node(pkg_path, batch_size);
    ros::spin();

    return 0;
}
