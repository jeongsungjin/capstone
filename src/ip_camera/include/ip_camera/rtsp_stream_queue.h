#ifndef RTSP_STREAM_QUEUE_H
#define RTSP_STREAM_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>

class RTSPStreamQueue {
public:
    explicit RTSPStreamQueue(const std::string& rtsp_url, size_t max_queue_size);
    ~RTSPStreamQueue();

    void start();
    void stop();

    std::vector<uint8_t> getFrame();
    // Timed wait version: returns empty vector if timeout expires
    std::vector<uint8_t> getFrameWaitFor(std::chrono::milliseconds timeout);
    // Non-blocking version: returns front frame or empty if none
    std::vector<uint8_t> getFrameNonBlocking();

private:
    void streamThread();

    std::string rtsp_url_;
    size_t max_queue_size_;

    std::queue<std::vector<uint8_t>> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::atomic<bool> running_{false};
    std::thread worker_thread_;
};

#endif // RTSP_STREAM_QUEUE_H