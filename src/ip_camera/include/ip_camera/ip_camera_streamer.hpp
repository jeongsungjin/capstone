#ifndef IP_CAMERA_STREAMER_HPP
#define IP_CAMERA_STREAMER_HPP

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <xmlrpcpp/XmlRpcValue.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <deque>
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
    ~IPCameraStreamer();

    void run();

private:
    // Threads
    void cameraThread(const CameraConfig cfg, int index);
    void publisherLoop();

    // Helpers
    std::vector<std::string> createStreamUrls(const CameraConfig& cfg) const;
    FILE* spawnFFmpeg(const std::string& url, int width, int height, const std::string& transport);

    void cleanup();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_{"~"};

    std::vector<CameraConfig> camera_configs_;
    std::vector<ros::Publisher> publishers_;

    // Latest frames per camera_id
    struct LatestFrame {
        cv::Mat frame;
        bool has{false};
    };

    std::mutex latest_mutex_;
    std::vector<LatestFrame> latest_; // indexed by camera_id-1

    std::atomic<bool> running_{true};
    std::vector<std::thread> cam_threads_;
    std::thread pub_thread_;

    double publish_rate_{60.0};
};

#endif // IP_CAMERA_STREAMER_HPP
