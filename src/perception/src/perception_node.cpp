#include "perception/perception_node.h"

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>

#include <chrono>

PerceptionNode::PerceptionNode(
    const std::string& pkg_path, 
    const std::string& image_topic_name,
    const int batch_size
): nh_("~"), perception_model_(pkg_path, batch_size)
{
    // 현재는 2개 이미지 동기화만 지원합니다.
    if (batch_size != 2) {
        ROS_WARN("PerceptionNode currently supports batch_size=2 for synchronized topics. Using first 2.");
    }

    // Initialize subscribers directly (non-copyable)
    const int queue_size = 5;
    image_sub1_.subscribe(nh_, image_topic_name + std::string("1/image_raw"), queue_size);
    image_sub2_.subscribe(nh_, image_topic_name + std::string("2/image_raw"), queue_size);

    // Synchronizer must be constructed with policy and subscribers
    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(queue_size), image_sub1_, image_sub2_);
    using namespace boost::placeholders;
    sync_->registerCallback(boost::bind(&PerceptionNode::imageCallback, this, _1, _2));

    // Visualization publishers (2 outputs)
    viz_result_pubs_.emplace_back(nh_.advertise<sensor_msgs::Image>("viz_result_1", 1));
    viz_result_pubs_.emplace_back(nh_.advertise<sensor_msgs::Image>("viz_result_2", 1));
}

void PerceptionNode::imageCallback(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    cv::Mat img1_mat = cv_ptr1->image;

    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
    cv::Mat img2_mat = cv_ptr2->image;

    std::vector<cv::Mat> img_batch;
    img_batch.emplace_back(img1_mat);
    img_batch.emplace_back(img2_mat);

    int ret = perception_model_.preprocess(img_batch);
    if (ret != 0) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return;
    }

    perception_model_.inference();
    perception_model_.postprocess();

    publishVizResult(img_batch);
}

// Removed unused publishBEVInfo

void PerceptionNode::publishVizResult(const std::vector<cv::Mat>& imgs){
    const auto& detections = perception_model_.getDetections();

    for(int b = 0; b < detections.size(); b++){
        for(int i = 0; i < detections[b].poly4s.size(); i++){
            cv::polylines(imgs[b], 
                std::vector<std::vector<cv::Point>>{
                    {
                        cv::Point(detections[b].poly4s[i](0, 0), detections[b].poly4s[i](0, 1)),
                        cv::Point(detections[b].poly4s[i](1, 0), detections[b].poly4s[i](1, 1)),
                        cv::Point(detections[b].poly4s[i](2, 0), detections[b].poly4s[i](2, 1)),
                        cv::Point(detections[b].poly4s[i](3, 0), detections[b].poly4s[i](3, 1))
                    }
                }, 
                true, 
                cv::Scalar(0, 255, 0), 
                2
            );
        }
        
        cv_bridge::CvImage out_msg;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = imgs[b];
        out_msg.header.stamp = ros::Time::now();
        viz_result_pubs_[b].publish(out_msg.toImageMsg());
    }
}

// Removed unused processing() path and thread-related code

PerceptionNode::~PerceptionNode(){}
