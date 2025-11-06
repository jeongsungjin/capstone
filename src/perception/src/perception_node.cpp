#include "perception/perception_node.h"

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include <boost/bind.hpp>
#include <boost/bind/placeholders.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <capstone_msgs/BEVInfo.h>

#include <chrono>

PerceptionNode::PerceptionNode(
    const std::string& pkg_path, 
    const std::string& image_topic_name,
    const int batch_size
): nh_("~"), perception_model_(pkg_path, batch_size)
{
    // 현재는 2개 이미지 동기화만 지원합니다.
    if (batch_size != 1) {
        ROS_WARN("PerceptionNode currently supports batch_size=2 for synchronized topics. Using first 2.");
    }

    image_sub_ = nh_.subscribe<sensor_msgs::Image>(
        image_topic_name + std::string("1/image_raw"), 
        1, 
        &PerceptionNode::imageCallback, 
        this
    );

    // Initialize subscribers directly (non-copyable)
    // const int queue_size = 5;
    // image_sub1_.subscribe(nh_, image_topic_name + std::string("1/image_raw"), queue_size);
    // image_sub2_.subscribe(nh_, image_topic_name + std::string("2/image_raw"), queue_size);

    // // Synchronizer must be constructed with policy and subscribers
    // sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(queue_size), image_sub1_, image_sub2_);
    // using namespace boost::placeholders;
    // sync_->registerCallback(boost::bind(&PerceptionNode::imageCallback, this, _1, _2));

    // Visualization publishers (2 outputs)

    bev_info_pub_ = nh_.advertise<capstone_msgs::BEVInfo>("bev_info", 1);
    viz_result_pubs_.emplace_back(nh_.advertise<sensor_msgs::Image>("viz_result_1", 1));
    viz_result_pubs_.emplace_back(nh_.advertise<sensor_msgs::Image>("viz_result_2", 1));
}

void PerceptionNode::imageCallback(const sensor_msgs::ImageConstPtr& img1){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img1_mat_ptr = std::make_shared<cv::Mat>(cv_ptr1->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img1_mat_ptr);

    int ret = perception_model_.preprocess(img_batch);
    if (ret != 0) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return;
    }

    perception_model_.inference();
    perception_model_.postprocess();

    publishBEVInfo();
    publishVizResult(img_batch);
}

void PerceptionNode::imageCallback(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img1_mat_ptr = std::make_shared<cv::Mat>(cv_ptr1->image);

    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img2_mat_ptr = std::make_shared<cv::Mat>(cv_ptr2->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img1_mat_ptr);
    img_batch.emplace_back(img2_mat_ptr);

    int ret = perception_model_.preprocess(img_batch);
    if (ret != 0) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return;
    }

    perception_model_.inference();
    perception_model_.postprocess();

    publishVizResult(img_batch);
}

void PerceptionNode::publishBEVInfo(){
    xt::xarray<float> H = {
        {99.001f, 212.3083f, -7019.0732f},
        {-656.1083f, 107.3403f, -2376.5612f},
        {-0.2113f, -0.4532f, -9.6068f}
    };

    auto Hinv = xt::linalg::inv(H);

    const auto& detections = perception_model_.getDetections();

    std::vector<size_t> ones_shape = {4, 1};

    capstone_msgs::BEVInfo bev_info;
    for(int b = 0; b < detections.size(); b++){
        for(auto& img_pts: detections[b].poly4s){
            auto ones = xt::ones<float>(ones_shape);
            
            auto homo_img_pts = xt::concatenate(
                xt::xtuple(
                    img_pts,
                    ones
                ), 1
            );

            auto homo_img_pts_T = xt::transpose(homo_img_pts);
            auto homo_world_pts = xt::linalg::dot(Hinv, homo_img_pts_T);
            auto world_pts = xt::view(homo_world_pts, xt::range(0, 2), xt::all()) / xt::view(homo_world_pts, 2, xt::all());
            // std::cout << world_pts << '\n';

            auto center_point = xt::mean(world_pts, {1});
            // std::cout << center_point << '\n';

            bev_info.detCounts += 1;
            bev_info.ids.emplace_back(10);
            bev_info.center_xs.emplace_back(center_point(0));
            bev_info.center_ys.emplace_back(center_point(1));
            bev_info.yaws.emplace_back(1.50);
        }
    }

    bev_info_pub_.publish(bev_info);
}

void PerceptionNode::publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs){
    const auto& detections = perception_model_.getDetections();

    for(int b = 0; b < detections.size(); b++){
        cv_bridge::CvImage out_msg;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        imgs[b]->copyTo(out_msg.image);
        out_msg.header.stamp = ros::Time::now();
        
        for(int i = 0; i < detections[b].poly4s.size(); i++){
            cv::polylines(out_msg.image, 
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
        
        viz_result_pubs_[b].publish(out_msg.toImageMsg());
    }
}

PerceptionNode::~PerceptionNode(){}
