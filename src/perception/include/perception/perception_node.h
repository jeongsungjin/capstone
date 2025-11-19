#ifndef PERCEPTION_NODE_H
#define PERCEPTION_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/int32.hpp>
#include <atomic>
#include <chrono>

#include "perception/model.h"

class PerceptionNode : public rclcpp::Node {
public:
    explicit PerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    void publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs);
    void publishBEVInfo();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    using ImgSubscriber = message_filters::Subscriber<ImageMsg>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void syncCallback(const ImageMsg::ConstSharedPtr& a, const ImageMsg::ConstSharedPtr& b,
                      const ImageMsg::ConstSharedPtr& c, const ImageMsg::ConstSharedPtr& d,
                      const ImageMsg::ConstSharedPtr& e, const ImageMsg::ConstSharedPtr& f);

    std::unique_ptr<ImgSubscriber> sub_a_, sub_b_, sub_c_, sub_d_, sub_e_, sub_f_;
    std::shared_ptr<Synchronizer> sync_;

    // model
    std::unique_ptr<Model> model_;

    std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> viz_result_pubs_;

    int batch_size_;
    std::string pkg_path_;

    // syncCallback rate monitoring
    std::atomic<uint64_t> sync_count_{0};
    std::chrono::steady_clock::time_point sync_start_;
};

#endif // PERCEPTION_NODE_H
