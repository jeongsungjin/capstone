#ifndef IP_CAMERA_STREAMER_HPP
#define IP_CAMERA_STREAMER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <cv_bridge/cv_bridge.hpp>

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
    explicit IPCameraStreamer();
    explicit IPCameraStreamer(const rclcpp::NodeOptions & options);
    ~IPCameraStreamer() override;

    void init();
    void stop();

private:
    void cameraThread();
    FILE* createCameraStream();

    CameraConfig camera_config_;
    std::string stream_url_;
    std::string ffmpeg_cmd_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;

    bool use_compression_{false};
    int jpeg_quality_{80};

    std::thread worker_;
    std::atomic<bool> running_{false};

    double publish_rate_{60.0};
    // publish timing stats
    std::atomic<uint64_t> publish_count_{0};
    std::atomic<uint64_t> total_publish_ns_{0};
    std::atomic<uint64_t> max_publish_ns_{0};
    std::chrono::steady_clock::time_point publish_stats_start_;
};

#endif
