#include "ip_camera/rtsp_stream_queue.h"
#include <iostream>
#include <cstdio>
#include <cstring>

RTSPStreamQueue::RTSPStreamQueue(const std::string& rtsp_url, size_t max_queue_size)
    : rtsp_url_(rtsp_url), max_queue_size_(max_queue_size) {}

RTSPStreamQueue::~RTSPStreamQueue() {
    stop();
}

void RTSPStreamQueue::start() {
    running_.store(true);
    worker_thread_ = std::thread(&RTSPStreamQueue::streamThread, this);
}

void RTSPStreamQueue::stop() {
    running_.store(false);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

std::vector<uint8_t> RTSPStreamQueue::getFrame() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this]() { return !frame_queue_.empty() || !running_.load(); });

    if (frame_queue_.empty()) {
        return {}; // Return an empty vector if stopped and no frames are available
    }

    auto frame = std::move(frame_queue_.front());
    frame_queue_.pop();
    return frame;
}

std::vector<uint8_t> RTSPStreamQueue::getFrameWaitFor(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    bool ready = queue_cv_.wait_for(lock, timeout, [this]() { return !frame_queue_.empty() || !running_.load(); });
    if (!ready || frame_queue_.empty()) {
        return {};
    }
    auto frame = std::move(frame_queue_.front());
    frame_queue_.pop();
    return frame;
}

std::vector<uint8_t> RTSPStreamQueue::getFrameNonBlocking() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (frame_queue_.empty()) {
        return {};
    }
    auto frame = std::move(frame_queue_.front());
    frame_queue_.pop();
    return frame;
}

// streamThread() replacement — reads MJPEG frames from ffmpeg image2pipe output
void RTSPStreamQueue::streamThread() {
    running_.store(true);

    // ffmpeg command: mjpeg per-frame -> image2pipe -> pipe:1
    // -rtsp_transport udp or tcp : 네트워크 환경에 따라 변경
    // -timeout 관련 옵션을 상황에 맞게 추가 가능
    std::string ffmpeg_cmd_base = 
        "ffmpeg -rtsp_transport udp -i \"" + rtsp_url_ + "\" "
        "-an -f image2pipe -vcodec mjpeg -q:v 5 pipe:1 2>/dev/null";

    std::cout << "rtsp_url_ : " << rtsp_url_ << std::endl;

    while (running_.load()) {
        FILE* pipe = popen(ffmpeg_cmd_base.c_str(), "r");
        if (!pipe) {
            std::cerr << "[RTSPStreamQueue] Failed to open ffmpeg pipe for: " << rtsp_url_ << std::endl;
            // 잠깐 쉬었다가 재시도
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        const size_t chunk_size = 64 * 1024; // 64KB 단위로 읽음
        std::vector<uint8_t> read_buf(chunk_size);
        std::vector<uint8_t> stream_buf; // 누적 버퍼

        while (running_.load()) {
            size_t n = fread(read_buf.data(), 1, chunk_size, pipe);
            if (n > 0) {
                // append
                stream_buf.insert(stream_buf.end(), read_buf.begin(), read_buf.begin() + n);

                // parse for JPEG frames: SOI = 0xFF 0xD8, EOI = 0xFF 0xD9
                // 반복해서 완전한 프레임을 추출
                size_t search_pos = 0;
                while (true) {
                    // 찾기: SOI
                    size_t soi = std::string::npos;
                    for (size_t i = search_pos; i + 1 < stream_buf.size(); ++i) {
                        if (stream_buf[i] == 0xFF && stream_buf[i+1] == 0xD8) {
                            soi = i;
                            break;
                        }
                    }
                    if (soi == std::string::npos) {
                        // SOI 못찾음 => stream_buf는 SOI 이전 바이트들을 버릴수도 있음
                        // stream_buf가 너무 크면 앞부분 자르기 (메모리 방지)
                        if (stream_buf.size() > 4 * 1024 * 1024) { // 4MB 임계
                            // SOI가 없으면 의미없는 바이트 절반 자르기
                            stream_buf.erase(stream_buf.begin(), stream_buf.begin() + stream_buf.size() / 2);
                        }
                        break;
                    }

                    // 찾기: EOI 이후
                    size_t eoi = std::string::npos;
                    for (size_t j = soi + 2; j + 1 < stream_buf.size(); ++j) {
                        if (stream_buf[j] == 0xFF && stream_buf[j+1] == 0xD9) {
                            eoi = j + 1; // eoi는 EOI 바이트 인덱스 (inclusive)
                            break;
                        }
                    }
                    if (eoi == std::string::npos) {
                        // EOI 아직 도착하지 않음 — 더 읽어야 함
                        // 하지만 buffer 앞부분에 SOI 이전 데이터가 쌓이면 메모리 증가하므로
                        // search_pos를 soi로 이동시켜 다음 탐색 효율화
                        search_pos = (soi + 2 < stream_buf.size()) ? soi + 2 : stream_buf.size();
                        break;
                    }

                    // 완전한 JPEG 프레임 (soi .. eoi) 추출
                    std::vector<uint8_t> jpeg_frame;
                    jpeg_frame.reserve(eoi - soi + 1);
                    jpeg_frame.insert(jpeg_frame.end(), stream_buf.begin() + soi, stream_buf.begin() + eoi + 1);

                    // 프레임을 큐에 넣기
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (frame_queue_.size() >= max_queue_size_) {
                            frame_queue_.pop();
                        }
                        frame_queue_.push(std::move(jpeg_frame));
                    }
                    queue_cv_.notify_one();

                    // stream_buf에서 추출한 부분 제거 (앞부분만 제거)
                    // 이때 남은 데이터는 eoi+1 부터 끝까지
                    stream_buf.erase(stream_buf.begin(), stream_buf.begin() + eoi + 1);

                    // 다음 프레임 검색은 버퍼 처음부터
                    search_pos = 0;
                }
            } else {
                // 읽은게 0인 경우: EOF 또는 에러
                if (feof(pipe)) {
                    std::cerr << "[RTSPStreamQueue] RTSP stream ended: " << rtsp_url_ << std::endl;
                } else if (ferror(pipe)) {
                    std::cerr << "[RTSPStreamQueue] Error reading ffmpeg pipe: " << strerror(errno) << " for " << rtsp_url_ << std::endl;
                }
                break;
            }
        } // inner while

        pclose(pipe);

        if (running_.load()) {
            // 연결이 끊겼지만 동작중이면 잠깐 대기 후 재연결 시도
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    } // outer while

    // thread 종료 시 notify 해서 대기중인 getFrame()이 빠져나가게 함
    queue_cv_.notify_all();
}
