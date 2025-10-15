#include <ros/ros.h>

#include "perception/perception_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "perception_node");
    PerceptionNode node;
    ros::spin();

    return 0;
}
