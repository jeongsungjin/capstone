#include "ip_camera/ip_camera_streamer.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <cstring>
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>
#include <cerrno>

using namespace std::chrono_literals;

IPCameraStreamer::IPCameraStreamer(const rclcpp::NodeOptions& options)
    : rclcpp::Node("ip_camera_streamer", options) 
{
    initConfig();
    this->declare_parameter<std::string>("topic_name_prefix", "/ipcam_");
    this->get_parameter("topic_name_prefix", camera_config_.topic_name_prefix);

    this->declare_parameter<int>("camera_id", 1);
    this->get_parameter("camera_id", camera_config_.camera_id);

    // create a simple rclcpp image publisher for now
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        camera_config_.topic_name_prefix + std::to_string(camera_config_.camera_id) + "/image_raw", 
        rclcpp::QoS(1));

    cv::setNumThreads(1);

    running_.store(true);
    worker_ = std::thread(&IPCameraStreamer::cameraThread, this);
}

void IPCameraStreamer::initConfig() {
    this->declare_parameter<std::string>("ip", "192.168.0.100");
    this->declare_parameter<int>("port", 554);
    this->declare_parameter<std::string>("username", "admin");
    this->declare_parameter<std::string>("password", "admin");
    this->declare_parameter<std::string>("topic_name", "image_raw");
    this->declare_parameter<std::string>("frame_id", "ipcam");
    this->declare_parameter<std::string>("transport", "tcp");
    this->declare_parameter<int>("width", 1280);
    this->declare_parameter<int>("height", 720);
    this->declare_parameter<double>("publish_rate", 30.0);

    this->get_parameter("port", camera_config_.port);
    this->get_parameter("username", camera_config_.username);
    this->get_parameter("password", camera_config_.password);
    this->get_parameter("width", camera_config_.width);
    this->get_parameter("height", camera_config_.height);
    this->get_parameter("transport", camera_config_.transport);
    
    this->get_parameter("ip", camera_config_.ip);
    this->get_parameter("frame_id", camera_config_.frame_id);

    this->get_parameter("publish_rate", publish_rate_);

    RCLCPP_INFO(
        this->get_logger(), 
        "[ip_camera] Initializing camera streamer for camera %d at %s (publish_rate=%.2f)",
        camera_config_.camera_id, camera_config_.ip.c_str(), publish_rate_);

    std::ostringstream url;
    url << "rtsp://" << camera_config_.username << ":" << camera_config_.password << "@"
        << camera_config_.ip << ":" << camera_config_.port << "/stream1";
    stream_url_ = url.str();

    std::ostringstream cmd;
    cmd << "ffmpeg -rtsp_transport " << camera_config_.transport << " "
        << "-probesize 5000000 -analyzeduration 10000000 "
        << "-fflags nobuffer -flags low_delay -i " << stream_url_ << " "
        << "-vf scale=" << camera_config_.width << ":" << camera_config_.height << " " 
        << "-pix_fmt bgr24 -f rawvideo -fps_mode cfr -r " << publish_rate_ << " "
        << "pipe:1 2>/dev/null";

    ffmpeg_cmd_ = cmd.str();
}

IPCameraStreamer::~IPCameraStreamer() { stop(); }

void IPCameraStreamer::stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
}

FILE* IPCameraStreamer::createCameraStream() {
    FILE* pipe = popen(ffmpeg_cmd_.c_str(), "r");
    return pipe;
}

void IPCameraStreamer::cameraThread() {
    const int width = camera_config_.width;
    const int height = camera_config_.height;
    const size_t bytes_per_frame = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;

    while (running_.load()) {
        FILE* proc = createCameraStream();
        if (!proc) {
            RCLCPP_ERROR(this->get_logger(), "[ip_camera] ffmpeg spawn failed for camera %d", camera_config_.camera_id);
            std::this_thread::sleep_for(10ms);
            continue;
        }

        auto start_time = std::chrono::steady_clock::now();
        int remaining = 0;

        while (running_.load()) {
            auto ros_now = this->get_clock()->now();

            auto img = std::make_unique<sensor_msgs::msg::Image>();

            img->width = static_cast<uint32_t>(width);
            img->height = static_cast<uint32_t>(height);
            img->step = static_cast<uint32_t>(width * 3);
            img->encoding = "bgr8";

            img->data.resize(bytes_per_frame);

            remaining = bytes_per_frame;
            unsigned char* write_ptr = img->data.data();
            while(remaining > 0) {
                size_t readn = fread(write_ptr, 1, remaining, proc);
                if (readn == 0) {
                    if (feof(proc)) {
                        RCLCPP_WARN(this->get_logger(), "[ip_camera] Camera %d stream EOF; will reconnect", camera_config_.camera_id);
                        break;
                    }

                    if (ferror(proc)) {
                        if (errno == EINTR) {
                            clearerr(proc);
                            continue;
                        }

                        RCLCPP_WARN(this->get_logger(), "[ip_camera] Camera %d read error: %s", camera_config_.camera_id, strerror(errno));
                        break;
                    }
                }

                remaining -= static_cast<int>(readn);
                write_ptr += readn;
            }

            if(remaining > 0) {
                RCLCPP_INFO(this->get_logger(), "[ip_camera] Incomplete frame read for camera %d; reconnecting", camera_config_.camera_id);
                break;
            }
            
            auto stamp = this->now();
            img->header.stamp = stamp;
            img->header.frame_id = camera_config_.frame_id;
            image_pub_->publish(std::move(img));
        }

        pclose(proc);
        std::this_thread::sleep_for(10ms);
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(IPCameraStreamer)
