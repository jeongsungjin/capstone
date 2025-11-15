// ROS2 multi-camera manager: creates batch_size IPCameraStreamer nodes and spins them

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>

#include "ip_camera/ip_camera_streamer.h"

static CameraConfig load_config_from_yaml(const std::string& cfg_path, int cam_id) {
    CameraConfig cfg;
    cfg.camera_id = cam_id;
    cfg.frame_id = "ipcam_" + std::to_string(cam_id);
    cfg.topic_name = "/ip_camera/" + std::to_string(cam_id) + "/image_raw";

    try {
        cv::FileStorage fs(cfg_path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            if (!fs["ip"].empty()) fs["ip"] >> cfg.ip;
            if (!fs["port"].empty()) fs["port"] >> cfg.port;
            if (!fs["username"].empty()) fs["username"] >> cfg.username;
            if (!fs["password"].empty()) fs["password"] >> cfg.password;
            if (!fs["frame_id"].empty()) fs["frame_id"] >> cfg.frame_id;
            if (!fs["transport"].empty()) fs["transport"] >> cfg.transport;
            if (!fs["width"].empty()) fs["width"] >> cfg.width;
            if (!fs["height"].empty()) fs["height"] >> cfg.height;
            if (!fs["topic_name"].empty()) fs["topic_name"] >> cfg.topic_name;
            fs.release();
        } else {
            RCLCPP_WARN(rclcpp::get_logger("ip_camera_node"),
                "Config %s not found or can't be opened; using defaults for camera %d", cfg_path.c_str(), cam_id);
        }
    } catch (const std::exception &e) {
        RCLCPP_WARN(rclcpp::get_logger("ip_camera_node"),
            "Error reading %s: %s; using defaults for camera %d", cfg_path.c_str(), e.what(), cam_id);
    }
    return cfg;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    int batch = 4;
    int offset = 0;
    std::string pkg_path = ".";

    if (argc > 1) { try { batch = std::stoi(argv[1]); } catch(...) {} }
    if (argc > 2) { try { offset = std::stoi(argv[2]); } catch(...) {} }
    if (argc > 3) {
        pkg_path = argv[3];
    } else if (const char* envp = std::getenv("CAPSTONE_ROOT")) {
        pkg_path = envp;
    }

    RCLCPP_INFO(rclcpp::get_logger("ip_camera_node"), "Starting with batch=%d offset=%d pkg_path=%s",
                batch, offset, pkg_path.c_str());

    std::vector<rclcpp::Node::SharedPtr> nodes;
    nodes.reserve(batch);

    for (int i = 0; i < batch; i++) {
        int cam_id = offset + i + 1;
        std::string cfg_path = pkg_path + "/src/ip_camera/config/ipcam_" + std::to_string(cam_id) + ".yaml";
        CameraConfig cfg = load_config_from_yaml(cfg_path, cam_id);

        auto node = std::make_shared<IPCameraStreamer>(cfg);
        nodes.emplace_back(node);
        RCLCPP_INFO(node->get_logger(), "Added camera id=%d ip=%s topic=%s", cfg.camera_id, cfg.ip.c_str(), cfg.topic_name.c_str());
    }

    rclcpp::executors::MultiThreadedExecutor exec;
    for (auto &n : nodes) exec.add_node(n);

    exec.spin();

    exec.cancel();
    nodes.clear();
    rclcpp::shutdown();
    return 0;
}
