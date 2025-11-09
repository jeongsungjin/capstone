#include "perception/perception_single_node.h"

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

PerceptionSingleNode::PerceptionSingleNode(
    const std::string& pkg_path, const int batch_size
): nh_("~"), H_(xt::zeros<double>({3, 3})), perception_model_(pkg_path, batch_size)
{
    std::string image_topic_prefix;
    ros::param::param<std::string>(
        "~image_topic_prefix", image_topic_prefix, "/camera/image_raw"
    );

    XmlRpc::XmlRpcValue H_param;
    nh_.getParam("/front_cam/H", H_param);
    for (int i = 0; i < 3; i++) {
        XmlRpc::XmlRpcValue row = H_param[i];
        for (int j = 0; j < 3; j++) {
            H_(i, j) = static_cast<double>(row[j]);

            std::cout << "H_(" << i << ", " << j << ") = " << H_(i, j) << std::endl;
        }
    }

    image_sub_ = nh_.subscribe<sensor_msgs::Image>(
        image_topic_prefix + std::string("1/image_raw"), 
        1, 
        &PerceptionSingleNode::imageCallback, 
        this
    );

    bev_info_pub_ = nh_.advertise<capstone_msgs::BEVInfo>("bev_info", 1);
    viz_result_pub_ = nh_.advertise<sensor_msgs::Image>("viz_result", 1);
}

void PerceptionSingleNode::imageCallback(const sensor_msgs::ImageConstPtr& img){
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img_mat_ptr = std::make_shared<cv::Mat>(cv_ptr->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img_mat_ptr);

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

void PerceptionSingleNode::publishBEVInfo(){
    const auto& detections = perception_model_.getDetections();

    capstone_msgs::BEVInfo bev_info;
    for(int b = 0; b < detections.size(); b++){
        for(auto& tri_pts: detections[b].tri_ptss){            
            auto center = xt::view(tri_pts, 0, xt::all());
            auto center3 = xt::concatenate(xt::xtuple(center, xt::xarray<double>({1.0})));

            auto bev_center = xt::linalg::dot(H_, center3);
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

void PerceptionSingleNode::publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs){
    const auto& detections = perception_model_.getDetections();

    cv_bridge::CvImage out_msg;
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    imgs[0]->copyTo(out_msg.image);
    out_msg.header.stamp = ros::Time::now();
    
    for(int i = 0; i < detections[0].poly4s.size(); i++){
        cv::polylines(out_msg.image, 
            std::vector<std::vector<cv::Point>>{{
                cv::Point(detections[0].poly4s[i](0, 0), detections[0].poly4s[i](0, 1)),
                cv::Point(detections[0].poly4s[i](1, 0), detections[0].poly4s[i](1, 1)),
                cv::Point(detections[0].poly4s[i](2, 0), detections[0].poly4s[i](2, 1)),
                cv::Point(detections[0].poly4s[i](3, 0), detections[0].poly4s[i](3, 1))
            }}, 
            true, 
            cv::Scalar(0, 255, 0), 
            2
        );
    }
    
    viz_result_pub_.publish(out_msg.toImageMsg());
}

PerceptionSingleNode::~PerceptionSingleNode(){}
