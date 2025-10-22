#include <ros/ros.h>
#include <ros/package.h>

#include "perception/perception_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "perception_node");
    
    std::string pkg_path = ros::package::getPath("perception");

    PerceptionNode node(pkg_path);
    ros::spin();

    return 0;
}
