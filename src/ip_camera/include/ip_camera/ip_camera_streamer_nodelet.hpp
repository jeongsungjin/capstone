#ifndef IP_CAMERA_STREAMER_NODELET_HPP
#define IP_CAMERA_STREAMER_NODELET_HPP

#include <nodelet/nodelet.h>
#include <memory>
#include "ip_camera/ip_camera_streamer.hpp"

namespace ip_camera {

class IPCameraStreamerNodelet : public nodelet::Nodelet {
public:
    IPCameraStreamerNodelet() = default;
    ~IPCameraStreamerNodelet() override = default;

private:
    void onInit() override;

    std::shared_ptr<IPCameraStreamer> streamer_;
};

} // namespace ip_camera

#endif
