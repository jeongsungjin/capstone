#ifndef IP_CAMERA_STREAMER_HPP
#define IP_CAMERA_STREAMER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <atomic>

struct CameraConfig {
    std::string ip;
    int port{554};
    std::string username;
    std::string password;
    std::string topic_name{"/image_raw"};
    std::string frame_id{"camera"};
    int camera_id{0};
    std::string transport{"tcp"}; // tcp or udp
    int width{1280};
    int height{720};
};

class IPCameraStreamer : public rclcpp::Node {
public:
    explicit IPCameraStreamer(const CameraConfig& camera_config);
    ~IPCameraStreamer() override;

    void stop();

private:
    void cameraThread();
    FILE* createCameraStream();

    CameraConfig camera_config_;
    std::string stream_url_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;

    std::thread worker_;
    std::atomic<bool> running_{false};
};

#endif
