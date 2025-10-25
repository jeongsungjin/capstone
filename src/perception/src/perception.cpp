#include <ros/ros.h>
#include <ros/package.h>

#include "perception/perception_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "perception_node");
    
    std::string pkg_path = ros::package::getPath("perception");

    std::string image_topic_name;
    ros::param::param<std::string>(
        "~image_topic", 
        image_topic_name, 
        "/camera/image_raw"
    );

    int batch_size;
    ros::param::param<int>(
        "~batch_size",
        batch_size,
        1
    );

    ROS_INFO_STREAM("ros param list");
    ROS_INFO_STREAM("image topic name : " << image_topic_name);
    ROS_INFO_STREAM("batch size : " << batch_size);    

    PerceptionNode node(pkg_path, image_topic_name, batch_size);
    ros::spin();

    return 0;
}
