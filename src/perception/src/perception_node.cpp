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

#include <yaml-cpp/yaml.h>

#include <chrono>

PerceptionNode::PerceptionNode(
    const std::string& pkg_path, const int batch_size
): nh_("~"), Hs_(xt::zeros<double>({batch_size, 3, 3})), perception_model_(pkg_path, batch_size)
{
    if (batch_size != 2 && batch_size != 4) {
        ROS_WARN("PerceptionNode currently supports batch_size=2 for synchronized topics. Using first 2.");
    }
    
    bev_info_pub_ = nh_.advertise<capstone_msgs::BEVInfo>("bev_info", 1);

    std::string topic_name_prefix;
    ros::param::param<std::string>(
        "~topic_name_prefix", topic_name_prefix, "/camera/image_raw"
    );

    const int queue_size = 5;
    std::vector<ros::Publisher>(batch_size).swap(viz_result_pubs_);
    std::vector<message_filters::Subscriber<sensor_msgs::Image>>(batch_size).swap(image_subs_);
    for(int b = 0; b < batch_size; b++){
        viz_result_pubs_[b] = nh_.advertise<sensor_msgs::Image>("viz_result_" + std::to_string(b + 1), 1);
        image_subs_[b].subscribe(nh_, topic_name_prefix + std::to_string(b + 1) + "/image_raw", queue_size);
    
        XmlRpc::XmlRpcValue H_param;
        nh_.getParam("/cam" + std::to_string(b + 1) + "/H", H_param);
        for (int i = 0; i < 3; i++) {
            XmlRpc::XmlRpcValue row = H_param[i];
            for (int j = 0; j < 3; j++) {
                Hs_(b, i, j) = static_cast<double>(row[j]);
                std::cout << "Hs_(" << b << ", " << i << ", " << j << ") = " << Hs_(b, i, j) << std::endl;
            }
        }
    }
    
    if(batch_size == 2){
        sync2_ = std::make_unique<message_filters::Synchronizer<SyncPolicy2>>(
            SyncPolicy2(queue_size), 
            image_subs_[0], 
            image_subs_[1]
        );

        using namespace boost::placeholders;
        sync2_->registerCallback(boost::bind(&PerceptionNode::imageCallback2, this, _1, _2));
    }
    
    else if(batch_size == 4){
        sync4_ = std::make_unique<message_filters::Synchronizer<SyncPolicy4>>(
            SyncPolicy4(queue_size), 
            image_subs_[0], 
            image_subs_[1],
            image_subs_[2],
            image_subs_[3]
        );

        using namespace boost::placeholders;
        sync4_->registerCallback(boost::bind(&PerceptionNode::imageCallback4, this, _1, _2, _3, _4));
    }
}

void PerceptionNode::imageCallback2(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img1_mat_ptr = std::make_shared<cv::Mat>(cv_ptr1->image);

    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img2_mat_ptr = std::make_shared<cv::Mat>(cv_ptr2->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img1_mat_ptr);
    img_batch.emplace_back(img2_mat_ptr);

    __processing(img_batch);
}

void PerceptionNode::imageCallback4(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2, const sensor_msgs::ImageConstPtr& img3, const sensor_msgs::ImageConstPtr& img4){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img1_mat_ptr = std::make_shared<cv::Mat>(cv_ptr1->image);

    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img2_mat_ptr = std::make_shared<cv::Mat>(cv_ptr2->image);

    cv_bridge::CvImagePtr cv_ptr3 = cv_bridge::toCvCopy(img3, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img3_mat_ptr = std::make_shared<cv::Mat>(cv_ptr3->image);

    cv_bridge::CvImagePtr cv_ptr4 = cv_bridge::toCvCopy(img4, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img4_mat_ptr = std::make_shared<cv::Mat>(cv_ptr4->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img1_mat_ptr);
    img_batch.emplace_back(img2_mat_ptr);
    img_batch.emplace_back(img3_mat_ptr);
    img_batch.emplace_back(img4_mat_ptr);

    __processing(img_batch);
}

void PerceptionNode::__processing(const std::vector<std::shared_ptr<cv::Mat>>& img_batch){
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
    const auto& detections = perception_model_.getDetections();

    capstone_msgs::BEVInfo bev_info;
    for(int b = 0; b < detections.size(); b++){
        auto H = xt::view(Hs_, b, xt::all(), xt::all());

        for(auto& tri_pts: detections[b].tri_ptss){            
            auto center = xt::view(tri_pts, 0, xt::all());
            auto center3 = xt::concatenate(xt::xtuple(center, xt::xarray<double>({1.0})));

            auto bev_center = xt::linalg::dot(H, center3);
            bev_center /= bev_center(2);

            bev_info.detCounts += 1;
            bev_info.ids.emplace_back(10);
            bev_info.center_xs.emplace_back(bev_center(0));
            bev_info.center_ys.emplace_back(bev_center(1));
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
                std::vector<std::vector<cv::Point>>{{
                    cv::Point(detections[b].poly4s[i](0, 0), detections[b].poly4s[i](0, 1)),
                    cv::Point(detections[b].poly4s[i](1, 0), detections[b].poly4s[i](1, 1)),
                    cv::Point(detections[b].poly4s[i](2, 0), detections[b].poly4s[i](2, 1)),
                    cv::Point(detections[b].poly4s[i](3, 0), detections[b].poly4s[i](3, 1))
                }}, 
                true, 
                cv::Scalar(0, 255, 0), 
                2
            );
        }
        
        viz_result_pubs_[b].publish(out_msg.toImageMsg());
    }
}

PerceptionNode::~PerceptionNode(){}
