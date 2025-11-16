#include "ip_camera/ip_camera_streamer.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <cstring>
#include <vector>

#include <rclcpp_components/register_node_macro.hpp>

using namespace std::chrono_literals;

IPCameraStreamer::IPCameraStreamer()
    : rclcpp::Node("ip_camera_streamer") { init(); }

IPCameraStreamer::IPCameraStreamer(const rclcpp::NodeOptions& options)
    : rclcpp::Node("ip_camera_streamer", options) { init(); }

IPCameraStreamer::~IPCameraStreamer() { stop(); }

void IPCameraStreamer::init(){
    this->declare_parameter<std::string>("ip", "192.168.0.100");
    this->declare_parameter<int>("port", 554);
    this->declare_parameter<std::string>("username", "admin");
    this->declare_parameter<std::string>("password", "admin");
    this->declare_parameter<std::string>("topic_name", "image_raw");
    this->declare_parameter<std::string>("frame_id", "ipcam");
    this->declare_parameter<int>("camera_id", 1);
    this->declare_parameter<std::string>("transport", "tcp");
    this->declare_parameter<int>("width", 1280);
    this->declare_parameter<int>("height", 720);
    this->declare_parameter<std::string>("topic_name_prefix", "/ipcam_");
    this->declare_parameter<double>("publish_rate", 60.0); // 0.0 = no throttle
    this->declare_parameter<std::string>("ffmpeg_hwaccel", ""); // e.g., "cuda", "vaapi"
    this->declare_parameter<int>("ffmpeg_threads", 0);           // 0 = ffmpeg default
    this->declare_parameter<double>("ffmpeg_fps", 0.0);           // 0.0 = no fps filter
    this->declare_parameter<bool>("use_compression", true);
    this->declare_parameter<int>("jpeg_quality", 80);
    
    this->get_parameter("port", camera_config_.port);
    this->get_parameter("username", camera_config_.username);
    this->get_parameter("password", camera_config_.password);
    this->get_parameter("topic_name_prefix", camera_config_.topic_name_prefix);
    this->get_parameter("width", camera_config_.width);
    this->get_parameter("height", camera_config_.height);
    this->get_parameter("transport", camera_config_.transport);
    this->get_parameter("publish_rate", publish_rate_);
    this->get_parameter("use_compression", use_compression_);
    this->get_parameter("jpeg_quality", jpeg_quality_);
    
    this->get_parameter("ip", camera_config_.ip);
    this->get_parameter("frame_id", camera_config_.frame_id);
    this->get_parameter("camera_id", camera_config_.camera_id);

    RCLCPP_INFO(
        this->get_logger(), 
        "[ip_camera] Initializing camera streamer for camera %d at %s (publish_rate=%.2f)",
        camera_config_.camera_id, camera_config_.ip.c_str(), publish_rate_);

    std::ostringstream url;
    url << "rtsp://" << camera_config_.username << ":" << camera_config_.password << "@"
        << camera_config_.ip << ":" << camera_config_.port << "/stream1";
    stream_url_ = url.str();

    std::ostringstream cmd;
    cmd << "ffmpeg "
        << "-rtsp_transport " << camera_config_.transport << " "
        << "-probesize 5000000 "
        << "-analyzeduration 10000000 "
        << "-fflags nobuffer "
        << "-flags low_delay "
        << "-i " << stream_url_ << " "
        << "-vf scale=1280:720 "
        << "-pix_fmt bgr24 "
        << "-f rawvideo "
        << "-fps_mode cfr "
        << "-r 30 "
        << "pipe:1 2>/dev/null";

    ffmpeg_cmd_ = cmd.str();
    // RCLCPP_INFO(this->get_logger(), "%s", ffmpeg_cmd_.c_str());


    // auto qos_profile = rclcpp::QoS(1);

    auto qos = rclcpp::SensorDataQoS();
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    qos.keep_last(10);
    image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        camera_config_.topic_name_prefix + std::to_string(camera_config_.camera_id) + "/image_raw", 
        qos
    );

    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        camera_config_.topic_name_prefix + std::to_string(camera_config_.camera_id) + "/image_raw/compressed",
        qos
    );

    cv::setNumThreads(1);

    running_.store(true);
    publish_count_.store(0);
    total_publish_ns_.store(0);
    max_publish_ns_.store(0);
    publish_stats_start_ = std::chrono::steady_clock::now();
    worker_ = std::thread(&IPCameraStreamer::cameraThread, this);
}

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
    rclcpp::Time last_pub(0, 0, this->get_clock()->get_clock_type());
    const double min_dt = (publish_rate_ > 0.0) ? (1.0 / publish_rate_) : 0.0;

    while (running_.load()) {
        FILE* proc = createCameraStream();
        if (!proc) {
            RCLCPP_ERROR(this->get_logger(), "[ip_camera] ffmpeg spawn failed for camera %d", camera_config_.camera_id);
            std::this_thread::sleep_for(500ms);
            continue;
        }

        auto start_time = std::chrono::steady_clock::now();
        int frame_count = 0;

        while (running_.load()) {
            auto ros_now = this->get_clock()->now();

            if (use_compression_) {
                // read raw frame into temporary buffer
                std::vector<uint8_t> raw_buf(bytes_per_frame);
                size_t readn = fread(raw_buf.data(), 1, bytes_per_frame, proc);
                if (readn != bytes_per_frame) {
                    RCLCPP_WARN(this->get_logger(), "[ip_camera] Camera %d stream interrupted; will reconnect", camera_config_.camera_id);
                    break;
                }

                // optional throttle if user set publish_rate_
                if (min_dt > 0.0 && (last_pub.nanoseconds() > 0) && ((ros_now.seconds() - last_pub.seconds()) < min_dt)) {
                    continue;
                }
                last_pub = ros_now;

                // convert raw_buf to cv::Mat and JPEG-encode
                cv::Mat mat(height, width, CV_8UC3, raw_buf.data());
                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
                std::vector<uchar> encoded;
                bool ok = cv::imencode(".jpg", mat, encoded, params);
                if (!ok) {
                    RCLCPP_WARN(this->get_logger(), "[ip_camera] JPEG encode failed for camera %d", camera_config_.camera_id);
                    continue;
                }

                sensor_msgs::msg::CompressedImage comp_msg;
                comp_msg.header.stamp = ros_now;
                comp_msg.header.frame_id = camera_config_.frame_id;
                comp_msg.format = "jpeg";
                comp_msg.data.assign(encoded.begin(), encoded.end());

                // measure publish latency
                auto pub_t0 = std::chrono::steady_clock::now();
                compressed_publisher_->publish(std::move(comp_msg));
                auto pub_t1 = std::chrono::steady_clock::now();
                uint64_t pub_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(pub_t1 - pub_t0).count());
                publish_count_.fetch_add(1, std::memory_order_relaxed);
                total_publish_ns_.fetch_add(pub_ns, std::memory_order_relaxed);
                uint64_t prev_max = max_publish_ns_.load(std::memory_order_relaxed);
                while (pub_ns > prev_max && !max_publish_ns_.compare_exchange_weak(prev_max, pub_ns, std::memory_order_relaxed)) {}

                frame_count++;
            } else {
                // Borrow a loaned message and read frame data directly into it to avoid extra copies
                auto loaned_msg = image_publisher_->borrow_loaned_message();
                sensor_msgs::msg::Image& msg = loaned_msg.get();

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

                // optional throttle if user set publish_rate_
                if (min_dt > 0.0 && (last_pub.nanoseconds() > 0) && ((ros_now.seconds() - last_pub.seconds()) < min_dt)) {
                    // skip publish to enforce rate; loaned_msg will be released on scope exit
                    continue;
                }
                last_pub = ros_now;

                msg.header.stamp = ros_now;
                msg.header.frame_id = camera_config_.frame_id;

                // measure publish latency
                auto pub_t0 = std::chrono::steady_clock::now();
                image_publisher_->publish(std::move(loaned_msg));
                auto pub_t1 = std::chrono::steady_clock::now();
                uint64_t pub_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(pub_t1 - pub_t0).count());
                publish_count_.fetch_add(1, std::memory_order_relaxed);
                total_publish_ns_.fetch_add(pub_ns, std::memory_order_relaxed);
                // update max
                uint64_t prev_max = max_publish_ns_.load(std::memory_order_relaxed);
                while (pub_ns > prev_max && !max_publish_ns_.compare_exchange_weak(prev_max, pub_ns, std::memory_order_relaxed)) {}

                frame_count++;
            }
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            if (elapsed.count() >= 1) {
                // publish timing stats
                uint64_t cnt = publish_count_.exchange(0);
                uint64_t total_ns = total_publish_ns_.exchange(0);
                uint64_t max_ns = max_publish_ns_.exchange(0);
                double avg_ms = (cnt > 0) ? (static_cast<double>(total_ns) / cnt) / 1e6 : 0.0;
                double max_ms = static_cast<double>(max_ns) / 1e6;

                RCLCPP_INFO(this->get_logger(), "[ip_camera] Camera %d FPS: %d | publish avg: %.3f ms max: %.3f ms", camera_config_.camera_id, frame_count, avg_ms, max_ms);
                frame_count = 0;
                start_time = now;
            }
        }

        pclose(proc);
        std::this_thread::sleep_for(500ms);
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(IPCameraStreamer)
