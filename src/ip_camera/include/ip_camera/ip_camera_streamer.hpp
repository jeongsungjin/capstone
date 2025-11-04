#ifndef IP_CAMERA_STREAMER_HPP
#define IP_CAMERA_STREAMER_HPP

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

struct CameraConfig {
    std::string ip;
    int port{554};
    std::string username;
    std::string password;
    std::string topic_name;
    std::string frame_id;
    int camera_id{0};
    std::string transport{"tcp"}; // tcp or udp
    int width{1280};
    int height{720};
};

class IPCameraStreamer {
public:
    IPCameraStreamer();
    IPCameraStreamer(ros::NodeHandle nh, ros::NodeHandle pnh);
    ~IPCameraStreamer();

    void run();
    void stop();

private:
    void cameraThread();
    FILE* spawnFFmpeg(const std::string& url, int width, int height, const std::string& transport);
    void cleanup();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_{"~"};

    CameraConfig camera_config_;
    std::string stream_url_;
    ros::Publisher img_publisher_;

    double publish_rate_{60.0};

    std::thread worker_;
    std::atomic<bool> running_{false};

    void init();
};

#endif
