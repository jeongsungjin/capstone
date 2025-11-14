#include "ip_camera/ip_camera_streamer.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <cstring>

using namespace std::chrono_literals;

IPCameraStreamer::IPCameraStreamer(const CameraConfig& camera_config)
    : rclcpp::Node("ip_camera_streamer_" + std::to_string(camera_config.camera_id))
    , camera_config_(camera_config) {
    // Build RTSP URL
    std::ostringstream oss;
    oss << "rtsp://" << camera_config_.username << ":" << camera_config_.password << "@"
        << camera_config_.ip << ":" << camera_config_.port << "/stream1";
    stream_url_ = oss.str();

    // Publisher with SensorData QoS
    image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        camera_config_.topic_name, rclcpp::SensorDataQoS());

    cv::setNumThreads(1);

    running_.store(true);
    auto* buf = new uint8_t[static_cast<size_t>(camera_config_.width) * static_cast<size_t>(camera_config_.height) * 3];
    worker_ = std::thread(&IPCameraStreamer::cameraThread, this, buf);
}

IPCameraStreamer::~IPCameraStreamer() {
    stop();
}

void IPCameraStreamer::stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
}

FILE* IPCameraStreamer::createCameraStream() {
    std::ostringstream cmd;
    std::string rtsp_transport = (camera_config_.transport == "udp" ? "udp" : "tcp");
    cmd << "ffmpeg "
        << "-rtsp_transport " << rtsp_transport << " "
        << "-fflags nobuffer -flags low_delay -probesize 16k -analyzeduration 0 "
        << "-i '" << stream_url_ << "' "
        << "-vf scale=" << camera_config_.width << ":" << camera_config_.height << " "
        << "-f rawvideo -pix_fmt bgr24 -vsync passthrough pipe:1 2>/dev/null";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    return pipe;
}

void IPCameraStreamer::cameraThread() {
    const int width = camera_config_.width;
    const int height = camera_config_.height;
    const size_t bytes_per_frame = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;

    while (running_.load()) {
        // reconnect loop
        FILE* proc = createCameraStream();
        if (!proc) {
            RCLCPP_ERROR(this->get_logger(), "[ip_camera] ffmpeg spawn failed for camera %d", camera_config_.camera_id);
            std::this_thread::sleep_for(500ms);
            continue;
        }

        while (running_.load()) {
            auto loaned_msg = image_publisher_->borrow_loaned_message();
            sensor_msgs::msg::Image& msg = loaned_msg.get();
            
            msg.header.stamp = this->now();
            msg.header.frame_id = camera_config_.frame_id;
            msg.width = static_cast<uint32_t>(width);
            msg.height = static_cast<uint32_t>(height);
            msg.step = static_cast<uint32_t>(width * 3);
            msg.encoding = "bgr8";
            msg.is_bigendian = false;
            
            msg.data.resize(bytes_per_frame);

            size_t readn = fread(msg.data.data(), 1, bytes_per_frame, proc);
            if (readn != bytes_per_frame) {
                RCLCPP_WARN(this->get_logger(), "[ip_camera] Camera %d stream interrupted; will reconnect", camera_config_.camera_id);
                break;
            }

            image_publisher_->publish(std::move(loaned_msg));
        }

        pclose(proc);
        std::this_thread::sleep_for(500ms);
    }
}
