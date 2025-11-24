#ifndef RTSP_STREAM_MANAGER_H
#define RTSP_STREAM_MANAGER_H

#include "rtsp_stream_queue.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp> // OpenCV 헤더 추가
#include <vector>

class RTSPStreamManager {
public:
    RTSPStreamManager(size_t max_streams = 7);
    ~RTSPStreamManager();

    bool addStream(const std::string& stream_id, const std::string& rtsp_url, cv::Mat map1, cv::Mat map2, size_t max_queue_size);
    bool removeStream(const std::string& stream_id);
    std::vector<uint8_t> getFrame(const std::string& stream_id);
    std::vector<std::shared_ptr<cv::Mat>> getAllFrames(); // batched nonblocking
    std::vector<std::shared_ptr<cv::Mat>> getAllFramesWithTimeout(std::chrono::milliseconds per_stream_timeout);

    void startAll();
    void stopAll();

private:
    size_t max_streams_;
    std::unordered_map<std::string, std::shared_ptr<RTSPStreamQueue>> streams_;
    // Preserve deterministic iteration order (e.g., ipcam_1..ipcam_7)
    std::vector<std::string> stream_order_;
    std::mutex manager_mutex_;
};

#endif // RTSP_STREAM_MANAGER_H