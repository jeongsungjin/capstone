#ifndef IP_CAMERA_STREAMER_HPP
#define IP_CAMERA_STREAMER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <memory>

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <atomic>

struct CameraConfig {
    std::string ip;
    int port{554};
    std::string username;
    std::string password;
    std::string topic_name_prefix{"/ipcam_"};
    std::string frame_id{"camera"};
    int camera_id{0};
    std::string transport{"tcp"}; // tcp or udp
    int width{1280};
    int height{720};
};

class IPCameraStreamer : public rclcpp::Node {
public:
    explicit IPCameraStreamer(const rclcpp::NodeOptions & options);
    ~IPCameraStreamer() override;

    void stop();

private:
    void initConfig();
    void cameraThread();
    FILE* createCameraStream();
    void createUndistortMap();

private:
    CameraConfig camera_config_;
    
    std::string stream_url_;
    std::string ffmpeg_cmd_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    double publish_rate_{30.0};

    cv::Mat K_, newK_, dist_;
    cv::Mat map1_, map2_;

    cv::Mat dst_mat_;

    std::thread worker_;
    std::atomic<bool> running_{false};
};

#endif
