#include "ip_camera/ip_camera_streamer.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <chrono>

using namespace std::chrono_literals;

IPCameraStreamer::IPCameraStreamer() : nh_(), pnh_("~") {
    // Load publish rate
    pnh_.param("publish_rate", publish_rate_, 60.0);

    // Try to load camera configs from ROS params: ~cameras (array of dicts)
    XmlRpc::XmlRpcValue cams;
    if (pnh_.getParam("cameras", cams) && cams.getType() == XmlRpc::XmlRpcValue::TypeArray && cams.size() > 0) {
        for (int i = 0; i < cams.size(); ++i) {
            if (cams[i].getType() != XmlRpc::XmlRpcValue::TypeStruct) {
                ROS_WARN("~cameras[%d] is not a struct, skipping", i);
                continue;
            }
            CameraConfig cfg;
            auto& m = cams[i];
            try {
                if (m.hasMember("ip")) cfg.ip = static_cast<std::string>(m["ip"]);
                if (m.hasMember("port")) cfg.port = static_cast<int>(m["port"]);
                if (m.hasMember("username")) cfg.username = static_cast<std::string>(m["username"]);
                if (m.hasMember("password")) cfg.password = static_cast<std::string>(m["password"]);
                if (m.hasMember("topic_name")) cfg.topic_name = static_cast<std::string>(m["topic_name"]);
                if (m.hasMember("frame_id")) cfg.frame_id = static_cast<std::string>(m["frame_id"]);
                if (m.hasMember("camera_id")) cfg.camera_id = static_cast<int>(m["camera_id"]);
                if (m.hasMember("transport")) cfg.transport = static_cast<std::string>(m["transport"]);
                if (m.hasMember("width")) cfg.width = static_cast<int>(m["width"]);
                if (m.hasMember("height")) cfg.height = static_cast<int>(m["height"]);
            } catch (...) {
                ROS_WARN("Error parsing ~cameras[%d]: XmlRpc exception", i);
            }

            if (cfg.topic_name.empty()) {
                // Create default topic from id if not provided
                std::ostringstream t;
                t << "/camera/camera_" << (cfg.camera_id > 0 ? cfg.camera_id : (i + 1)) << "/image_raw";
                cfg.topic_name = t.str();
            }
            if (cfg.frame_id.empty()) {
                std::ostringstream f;
                f << "camera_" << (cfg.camera_id > 0 ? cfg.camera_id : (i + 1)) << "_link";
                cfg.frame_id = f.str();
            }
            if (cfg.camera_id <= 0) cfg.camera_id = i + 1;
            if (cfg.ip.empty()) {
                ROS_WARN("~cameras[%d] missing ip, skipping", i);
                continue;
            }
            camera_configs_.push_back(cfg);
        }
    }

    // Fallback to hardcoded defaults if no params
    if (camera_configs_.empty()) {
        ROS_WARN("~cameras not set or empty. Falling back to two hardcoded cameras.");
        camera_configs_.push_back(CameraConfig{ "192.168.0.20", 554, "admin", "zjsxmfhf", "/camera/camera_1/image_raw", "camera_1_link", 1, "tcp", 1280, 720 });
        camera_configs_.push_back(CameraConfig{ "192.168.0.21", 554, "admin", "zjsxmfhf", "/camera/camera_2/image_raw", "camera_2_link", 2, "tcp", 1280, 720 });
    }

    // Prepare publishers and latest buffers
    publishers_.reserve(camera_configs_.size());
    for (const auto& cfg : camera_configs_) {
        publishers_.push_back(nh_.advertise<sensor_msgs::Image>(cfg.topic_name, 1));
    }
    latest_.assign(camera_configs_.size(), LatestFrame{});

    // Start camera threads with index mapping
    for (size_t i = 0; i < camera_configs_.size(); ++i) {
        cam_threads_.emplace_back(&IPCameraStreamer::cameraThread, this, camera_configs_[i], static_cast<int>(i));
    }

    // Publisher thread
    pub_thread_ = std::thread(&IPCameraStreamer::publisherLoop, this);

    ROS_INFO("IP Camera Streamer initialized (%zu cameras)", camera_configs_.size());
    for (size_t i = 0; i < camera_configs_.size(); ++i) {
        ROS_INFO("  Camera %d: %s -> %s", camera_configs_[i].camera_id, camera_configs_[i].ip.c_str(), camera_configs_[i].topic_name.c_str());
    }

    // OpenCV optimization
    cv::setNumThreads(1);
}

IPCameraStreamer::~IPCameraStreamer() {
    cleanup();
}

std::vector<std::string> IPCameraStreamer::createStreamUrls(const CameraConfig& cfg) const {
    std::vector<std::string> urls;
    std::ostringstream s1, s2;
    s1 << "rtsp://" << cfg.username << ":" << cfg.password << "@" << cfg.ip << ":" << cfg.port << "/stream1";
    s2 << "rtsp://" << cfg.username << ":" << cfg.password << "@" << cfg.ip << ":" << cfg.port << "/stream2";
    urls.push_back(s1.str());
    urls.push_back(s2.str());
    return urls;
}

// Spawn ffmpeg process that outputs raw BGR24 frames to stdout (pipe)
FILE* IPCameraStreamer::spawnFFmpeg(const std::string& url, int width, int height, const std::string& transport) {
    std::ostringstream cmd;
    std::string rtsp_transport = (transport == "udp" ? "udp" : "tcp");
    cmd 
        << "ffmpeg "
        << "-rtsp_transport " << rtsp_transport << " "
        << "-fflags nobuffer -flags low_delay -probesize 16k -analyzeduration 0 "
        << "-i '" << url << "' "
        << "-vf scale=" << width << ":" << height << " "
        << "-f rawvideo -pix_fmt bgr24 -vsync passthrough pipe:1 2>/dev/null";

    // Use popen to read stdout
    FILE* pipe = popen(cmd.str().c_str(), "r");
    return pipe; // nullptr on failure
}

void IPCameraStreamer::cameraThread(const CameraConfig cfg, int cam_index) {
    if (cam_index < 0 || cam_index >= static_cast<int>(camera_configs_.size())) {
        ROS_ERROR("Invalid camera index=%d", cam_index);
        return;
    }

    ROS_INFO("Camera %d connection attempt start", cfg.camera_id);
    const auto urls = createStreamUrls(cfg);

    FILE* proc = nullptr;
    int width = cfg.width;
    int height = cfg.height;
    const size_t bytes_per_frame = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    std::vector<unsigned char> buffer(bytes_per_frame);

    // Try URLs in order
    for (size_t i = 0; i < urls.size() && running_.load(); ++i) {
        ROS_INFO("Camera %d trying URL (%zu/%zu): %s", cfg.camera_id, i + 1, urls.size(), urls[i].c_str());
        proc = spawnFFmpeg(urls[i], width, height, cfg.transport);
        if (!proc) {
            ROS_ERROR("ffmpeg spawn failed for camera %d", cfg.camera_id);
            std::this_thread::sleep_for(500ms);
            continue;
        }

        // Test read one frame
        size_t readn = fread(buffer.data(), 1, bytes_per_frame, proc);
        if (readn == bytes_per_frame) {
            cv::Mat frame(height, width, CV_8UC3, buffer.data());
            {
                std::lock_guard<std::mutex> lk(latest_mutex_);
                latest_[cam_index].frame = frame.clone();
                latest_[cam_index].has = true;
            }
            ROS_INFO("✅ Camera %d connected: %s (%dx%d)", cfg.camera_id, urls[i].c_str(), width, height);
            break;
        } else {
            ROS_WARN("❌ Camera %d URL failed (no frame): %s", cfg.camera_id, urls[i].c_str());
            pclose(proc);
            proc = nullptr;
            std::this_thread::sleep_for(500ms);
        }
    }

    if (!proc) {
        ROS_ERROR("Camera %d connection failed - all URLs tried", cfg.camera_id);
        return;
    }

    // Streaming loop
    size_t frame_count = 0;
    auto last = std::chrono::steady_clock::now();
    while (running_.load() && ros::ok()) {
        size_t readn = fread(buffer.data(), 1, bytes_per_frame, proc);
        if (readn != bytes_per_frame) {
            ROS_WARN("Camera %d stream ended/interrupted, reconnecting...", cfg.camera_id);
            pclose(proc);
            proc = nullptr;
            // break to outer reconnection logic
            break;
        }

        cv::Mat frame(height, width, CV_8UC3, buffer.data());
        {
            std::lock_guard<std::mutex> lk(latest_mutex_);
            latest_[cam_index].frame = frame.clone();
            latest_[cam_index].has = true;
        }

        frame_count++;
        auto now = std::chrono::steady_clock::now();
        if (now - last >= 5s) {
            double fps = frame_count / std::chrono::duration<double>(now - last).count();
            ROS_INFO("Camera %d capture fps≈%.2f", cfg.camera_id, fps);
            frame_count = 0;
            last = now;
        }
    }

    if (proc) {
        pclose(proc);
    }
    ROS_INFO("Camera %d streaming thread exit", cfg.camera_id);
}

void IPCameraStreamer::publisherLoop() {
    ros::Rate rate(publish_rate_ > 1e-3 ? publish_rate_ : 60.0);
    while (running_.load() && ros::ok()) {
        ros::Time stamp = ros::Time::now();

        // Publish latest frames with same stamp
        for (size_t i = 0; i < camera_configs_.size(); ++i) {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lk(latest_mutex_);
                if (i >= latest_.size() || !latest_[i].has) continue;
                frame = latest_[i].frame; // copy header
            }
            if (!frame.empty()) {
                try {
                    cv_bridge::CvImage out;
                    out.header.stamp = stamp;
                    out.header.frame_id = camera_configs_[i].frame_id;
                    out.encoding = "bgr8";
                    out.image = frame;
                    publishers_[i].publish(out.toImageMsg());
                } catch (const std::exception& e) {
                    ROS_WARN("Camera %d publish error: %s", camera_configs_[i].camera_id, e.what());
                }
            }
        }

        rate.sleep();
    }
}

void IPCameraStreamer::cleanup() {
    if (!running_.exchange(false)) return; // already stopped

    for (auto& th : cam_threads_) {
        if (th.joinable()) th.join();
    }
    if (pub_thread_.joinable()) {
        pub_thread_.join();
    }
}

void IPCameraStreamer::run() {
    try {
        ros::spin();
    } catch (...) {
        ROS_INFO("IPCameraStreamer shutting down due to exception");
    }
    cleanup();
}
