#ifndef __PERCEPTION_SINGLE_NODE_H__
#define __PERCEPTION_SINGLE_NODE_H__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>

#include "Model.h"

#include <xtensor/xarray.hpp>

#include <vector>
#include <memory>

class PerceptionSingleNode{
public:
    PerceptionSingleNode(const std::string& pkg_path, const int batch_size);
    ~PerceptionSingleNode();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& img);
    void publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs);
    void publishBEVInfo();

private:
    ros::NodeHandle nh_;
    
    ros::Subscriber image_sub_;    
    ros::Publisher bev_info_pub_, viz_result_pub_;
    
    Model perception_model_;

    xt::xarray<double> H_;
};

#endif