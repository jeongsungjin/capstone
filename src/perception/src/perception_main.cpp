#include "perception/perception_node.h"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PerceptionNode>(rclcpp::NodeOptions());

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}