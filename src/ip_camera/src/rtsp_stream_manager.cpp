#include "ip_camera/rtsp_stream_manager.h"
#include <opencv2/opencv.hpp> // OpenCV 헤더 추가
#include <chrono>

RTSPStreamManager::RTSPStreamManager(size_t max_streams) : max_streams_(max_streams) {}

RTSPStreamManager::~RTSPStreamManager() {
    stopAll();
}

bool RTSPStreamManager::addStream(const std::string& stream_id, const std::string& rtsp_url, cv::Mat map1, cv::Mat map2, size_t max_queue_size) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    if (streams_.size() >= max_streams_) {
        return false; // 최대 스트림 수 초과
    }
    if (streams_.find(stream_id) != streams_.end()) {
        return false; // 이미 존재하는 스트림 ID
    }

    auto stream = std::make_shared<RTSPStreamQueue>(rtsp_url, map1, map2, max_queue_size);
    streams_[stream_id] = stream;
    stream_order_.push_back(stream_id);
    stream->start();
    return true;
}

bool RTSPStreamManager::removeStream(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return false; // 스트림 ID를 찾을 수 없음
    }

    it->second->stop();
    streams_.erase(it);
    // remove from order list
    stream_order_.erase(std::remove(stream_order_.begin(), stream_order_.end(), stream_id), stream_order_.end());
    return true;
}

std::vector<uint8_t> RTSPStreamManager::getFrame(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        return {}; // 스트림 ID를 찾을 수 없음
    }

    return it->second->getFrame();
}

void RTSPStreamManager::startAll() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (const auto& id : stream_order_) {
        auto it = streams_.find(id);
        if (it != streams_.end()) it->second->start();
    }
}

void RTSPStreamManager::stopAll() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    for (const auto& id : stream_order_) {
        auto it = streams_.find(id);
        if (it != streams_.end()) it->second->stop();
    }
}

std::vector<std::shared_ptr<cv::Mat>> RTSPStreamManager::getAllFrames() {
    // Copy shared_ptrs under lock then release to avoid holding while blocking
    std::vector<std::shared_ptr<RTSPStreamQueue>> queues;
    {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        queues.reserve(streams_.size());
        for (const auto& id : stream_order_) {
            auto it = streams_.find(id);
            if (it != streams_.end()) queues.push_back(it->second);
        }
    }
    std::vector<std::shared_ptr<cv::Mat>> frames;
    frames.reserve(queues.size());
    for (auto &q : queues) {
        auto frame_data = q->getFrameNonBlocking();
        if (frame_data.empty()) {
            continue;
        }
        cv::Mat decoded = cv::imdecode(frame_data, cv::IMREAD_COLOR);
        if (!decoded.empty()) {
            frames.emplace_back(std::make_shared<cv::Mat>(std::move(decoded)));
        }
    }
    return frames;
}

std::vector<std::shared_ptr<cv::Mat>> RTSPStreamManager::getAllFramesWithTimeout(std::chrono::milliseconds per_stream_timeout) {
    std::vector<std::shared_ptr<RTSPStreamQueue>> queues;
    {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        queues.reserve(streams_.size());
        for (const auto& id : stream_order_) {
            auto it = streams_.find(id);
            if (it != streams_.end()) queues.push_back(it->second);
        }
    }
    std::vector<std::shared_ptr<cv::Mat>> frames;
    frames.reserve(queues.size());
    for (auto &q : queues) {
        auto frame_data = q->getFrameWaitFor(per_stream_timeout);
        if (frame_data.empty()) {
            continue; // timeout or no frame
        }
        cv::Mat decoded = cv::imdecode(frame_data, cv::IMREAD_COLOR);
        cv::resize(decoded, decoded, cv::Size(1536, 864));
        
        // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
        // std::cout << "Decoded frame size: " << decoded.cols << "x" << decoded.rows << std::endl;

        if (!decoded.empty()) {
            cv::Mat undistorted;
            cv::remap(decoded, undistorted, q->map1_, q->map2_, cv::INTER_LINEAR);
            frames.emplace_back(std::make_shared<cv::Mat>(std::move(undistorted)));
        }
    }
    return frames;
}