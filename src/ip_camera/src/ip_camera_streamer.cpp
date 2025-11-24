#include "ip_camera/ip_camera_streamer.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <cstring>
#include <vector>

#include <memory>

#include <rclcpp_components/register_node_macro.hpp>
#include <cerrno>

using namespace std::chrono_literals;

IPCameraStreamer::IPCameraStreamer(const rclcpp::NodeOptions& options)
    : rclcpp::Node("ip_camera_streamer", options) 
{
    this->declare_parameter<std::string>("topic_name_prefix", "/ipcam_");
    this->get_parameter("topic_name_prefix", camera_config_.topic_name_prefix);
    
    this->declare_parameter<int>("camera_id", 1);
    this->get_parameter("camera_id", camera_config_.camera_id);
    
    // create a simple rclcpp image publisher for now
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        camera_config_.topic_name_prefix + std::to_string(camera_config_.camera_id) + "/image_raw", 
        rclcpp::QoS(1));
    
    initConfig();

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

    this->declare_parameter<std::vector<double>>("K", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("new_K", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("dist", std::vector<double>{});

    // Read parameter vectors and convert to cv::Mat
    std::vector<double> K_vec, newK_vec, dist_vec;
    this->get_parameter("K", K_vec);
    this->get_parameter("new_K", newK_vec);
    this->get_parameter("dist", dist_vec);

    if (!K_vec.empty()) {
        if (K_vec.size() == 9) {
            K_ = cv::Mat(3, 3, CV_64F);
            std::memcpy(K_.ptr<double>(), K_vec.data(), sizeof(double) * 9);
        } else {
            RCLCPP_WARN(this->get_logger(), "Parameter K has unexpected size %zu (expected 9)", K_vec.size());
        }
    }

    if (!newK_vec.empty()) {
        if (newK_vec.size() == 9) {
            newK_ = cv::Mat(3, 3, CV_64F);
            std::memcpy(newK_.ptr<double>(), newK_vec.data(), sizeof(double) * 9);
        } else {
            RCLCPP_WARN(this->get_logger(), "Parameter new_K has unexpected size %zu (expected 9)", newK_vec.size());
        }
    }

    if (!dist_vec.empty()) {
        // distortion coefficients can be e.g. 4,5,8 elements; store as 1xN CV_64F
        dist_ = cv::Mat(1, static_cast<int>(dist_vec.size()), CV_64F);
        std::memcpy(dist_.ptr<double>(), dist_vec.data(), sizeof(double) * dist_vec.size());
    }

    // read remaining runtime parameters (width/height must be known before building maps)
    this->get_parameter("port", camera_config_.port);
    this->get_parameter("username", camera_config_.username);
    this->get_parameter("password", camera_config_.password);
    this->get_parameter("width", camera_config_.width);
    this->get_parameter("height", camera_config_.height);
    this->get_parameter("transport", camera_config_.transport);
    
    this->get_parameter("ip", camera_config_.ip);
    this->get_parameter("frame_id", camera_config_.frame_id);

    this->get_parameter("publish_rate", publish_rate_);

    // build undistort/remap maps now that width/height (and intrinsics) are final
    createUndistortMap();

    dst_mat_ = cv::Mat(camera_config_.height, camera_config_.width, CV_8UC3);

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
        << "-probesize 32k -analyzeduration 10000000 "
        << "-fflags nobuffer+discardcorrupt -flags low_delay -i " << stream_url_ << " "
        << "-vf scale=" << camera_config_.width << ":" << camera_config_.height << " " 
        << "-pix_fmt bgr24 -f rawvideo -fps_mode cfr -r " << publish_rate_ << " "
        << "pipe:1 2>/dev/null";

    ffmpeg_cmd_ = cmd.str();
}

IPCameraStreamer::~IPCameraStreamer() { stop(); }

void IPCameraStreamer::createUndistortMap(){
    cv::Mat newCameraMatrix = newK_.empty() ? K_ : newK_;

    cv::initUndistortRectifyMap(
        K_,                                                     // 원본 카메라 행렬
        dist_,                                                  // 왜곡 계수
        cv::Mat(),                                              // R (아이덴티티 행렬)
        newCameraMatrix,                                        // 새로운 카메라 행렬 (출력 영상용)
        cv::Size(camera_config_.width, camera_config_.height),  // 출력 이미지 크기
        CV_32FC1,                                               // map 타입
        map1_, map2_
    );
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

    std::vector<uint8_t> raw_buffer(bytes_per_frame);

    while (running_.load()) {
        FILE* proc = createCameraStream();
        if (!proc) {
            RCLCPP_ERROR(this->get_logger(), "[ip_camera] ffmpeg spawn failed for camera %d", camera_config_.camera_id);
            std::this_thread::sleep_for(10ms);
            continue;
        }

        auto start_time = std::chrono::steady_clock::now();

        while (running_.load()) {
            size_t remaining = bytes_per_frame;
            unsigned char* write_ptr = raw_buffer.data();

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
            
            // ---- REMAP ----
            cv::Mat src(height, width, CV_8UC3, raw_buffer.data());

            cv::resize(src, src, cv::Size(1536, 864));
            cv::remap(src, dst_mat_, map1_, map2_, cv::INTER_LINEAR);

            // ---- PUBLISH RAW BGR IMAGE ----
            // Use remapped image (dst_mat_) dimensions
            const int out_width = dst_mat_.cols;
            const int out_height = dst_mat_.rows;
            auto img = std::make_unique<sensor_msgs::msg::Image>();
            img->header.stamp = this->now();
            img->header.frame_id = camera_config_.frame_id;
            img->height = out_height;
            img->width = out_width;
            img->encoding = sensor_msgs::image_encodings::BGR8;
            img->is_bigendian = false;
            img->step = static_cast<sensor_msgs::msg::Image::_step_type>(out_width * 3);
            const size_t data_size = static_cast<size_t>(out_height) * static_cast<size_t>(out_width) * 3;
            img->data.resize(data_size);
            std::memcpy(img->data.data(), dst_mat_.data, data_size);
            image_pub_->publish(std::move(img));
        }

        pclose(proc);
        std::this_thread::sleep_for(10ms);
    }

}

RCLCPP_COMPONENTS_REGISTER_NODE(IPCameraStreamer)
