#ifndef PERCEPTION_NODELET_HPP
#define PERCEPTION_NODELET_HPP

#include <nodelet/nodelet.h>
#include <memory>
#include <ros/ros.h>

#include "perception/perception_node.h"

namespace perception_ns {

class PerceptionNodelet : public nodelet::Nodelet {
public:
    PerceptionNodelet() = default;
    ~PerceptionNodelet() override = default;

private:
    void onInit() override;

    std::unique_ptr<PerceptionNode> node_;
};

} // namespace perception_ns

#endif // PERCEPTION_NODELET_HPP
