#include "ip_camera/ip_camera_streamer.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>

using namespace std::chrono_literals;

IPCameraStreamer::IPCameraStreamer() : nh_(), pnh_("~") {
    init();
}

IPCameraStreamer::IPCameraStreamer(ros::NodeHandle nh, ros::NodeHandle pnh) : nh_(nh), pnh_(pnh) {
    init();
}

IPCameraStreamer::~IPCameraStreamer() {
    cleanup();
}

void IPCameraStreamer::init() {
    pnh_.param("publish_rate", publish_rate_, 60.0);
    pnh_.getParam("ip", camera_config_.ip);
    pnh_.param("port", camera_config_.port, camera_config_.port);
    pnh_.param("username", camera_config_.username, std::string());
    pnh_.param("password", camera_config_.password, std::string());
    pnh_.param("topic_name", camera_config_.topic_name, std::string());
    pnh_.param("frame_id", camera_config_.frame_id, std::string());
    pnh_.param("camera_id", camera_config_.camera_id, 1);
    pnh_.param("transport", camera_config_.transport, std::string("tcp"));
    pnh_.param("width", camera_config_.width, camera_config_.width);
    pnh_.param("height", camera_config_.height, camera_config_.height);

    bool valid = true;
    if (camera_config_.ip.empty()) {
        ROS_ERROR("[ip_camera] Param '~ip' is required but empty");
        valid = false;
    }
    if (camera_config_.topic_name.empty()) {
        ROS_ERROR("[ip_camera] Param '~topic_name' is required but empty");
        valid = false;
    }
    if (camera_config_.frame_id.empty()) {
        ROS_ERROR("[ip_camera] Param '~frame_id' is required but empty");
        valid = false;
    }
    if (camera_config_.width <= 0 || camera_config_.height <= 0) {
        ROS_ERROR("[ip_camera] Params '~width' and '~height' must be > 0");
        valid = false;
    }

    if (!valid) {
        ROS_ERROR("[ip_camera] Invalid configuration. Streamer will not start.");
        return;
    }

    img_publisher_ = nh_.advertise<sensor_msgs::Image>(camera_config_.topic_name, 1);

    std::ostringstream oss;
    oss << "rtsp://" << camera_config_.username << ":" << camera_config_.password << "@"
        << camera_config_.ip << ":" << camera_config_.port << "/stream1";
    stream_url_ = oss.str();

    ROS_INFO("[ip_camera] id=%d ip=%s transport=%s size=%dx%d topic=%s frame_id=%s",
             camera_config_.camera_id,
             camera_config_.ip.c_str(),
             camera_config_.transport.c_str(),
             camera_config_.width,
             camera_config_.height,
             camera_config_.topic_name.c_str(),
             camera_config_.frame_id.c_str());

    cv::setNumThreads(1);

    running_.store(true);
    worker_ = std::thread(&IPCameraStreamer::cameraThread, this);
}

FILE* IPCameraStreamer::spawnFFmpeg(const std::string& url, int width, int height, const std::string& transport) {
    std::ostringstream cmd;
    std::string rtsp_transport = (transport == "udp" ? "udp" : "tcp");
    cmd << "ffmpeg "
        << "-rtsp_transport " << rtsp_transport << " "
        << "-fflags nobuffer -flags low_delay -probesize 16k -analyzeduration 0 "
        << "-i '" << url << "' "
        << "-vf scale=" << width << ":" << height << " "
        << "-f rawvideo -pix_fmt bgr24 -vsync passthrough pipe:1 2>/dev/null";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    return pipe;
}

void IPCameraStreamer::cameraThread() {
    const int width = camera_config_.width;
    const int height = camera_config_.height;
    const size_t bytes_per_frame = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    std::vector<unsigned char> buffer(bytes_per_frame);

    ros::Time last_pub = ros::Time(0);
    const double min_dt = (publish_rate_ > 0.0) ? (1.0 / publish_rate_) : 0.0;

    while (running_.load() && ros::ok()) {
        // -> 재연결 로직
        FILE* proc = spawnFFmpeg(stream_url_, width, height, camera_config_.transport);
        if (!proc) {
            ROS_ERROR("[ip_camera] ffmpeg spawn failed for camera %d", camera_config_.camera_id);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        // -> 프레임 읽기 및 퍼블리시
        while (running_.load() && ros::ok()) {
            size_t readn = fread(buffer.data(), 1, bytes_per_frame, proc);
            if (readn != bytes_per_frame) {
                // -> 문제가 생기면 다시 읽기
                ROS_WARN("[ip_camera] Camera %d stream interrupted; will reconnect", camera_config_.camera_id);
                break;
            }

            ros::Time now = ros::Time::now();
            if (min_dt > 0.0 && (last_pub.toSec() > 0.0) && ((now - last_pub).toSec() < min_dt)) {
                continue;
            }

            last_pub = now;

            cv::Mat frame(height, width, CV_8UC3, buffer.data());
            try {
                cv_bridge::CvImage out;
                out.header.stamp = now;
                out.header.frame_id = camera_config_.frame_id;
                out.encoding = "bgr8";
                out.image = frame;
                img_publisher_.publish(out.toImageMsg());
            } catch (const std::exception& e) {
                ROS_WARN("[ip_camera] Camera %d publish error: %s", camera_config_.camera_id, e.what());
            }
        }

        pclose(proc);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void IPCameraStreamer::run() {
    ros::spin();
    cleanup();
}

void IPCameraStreamer::stop() {
    cleanup();
}

void IPCameraStreamer::cleanup() {
    if (!running_.exchange(false)) return;
    if (worker_.joinable()) worker_.join();
}
