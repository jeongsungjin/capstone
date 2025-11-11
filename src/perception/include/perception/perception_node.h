#ifndef __PERCEPTION_NODE_H__
#define __PERCEPTION_NODE_H__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>

#include "Model.h"

#include <vector>
#include <memory>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy2;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SyncPolicy4;

class PerceptionNode{
public:
    PerceptionNode(ros::NodeHandle nh, ros::NodeHandle pnh, const std::string& pkg_path, const int batch_size);
    ~PerceptionNode();

private:
    void imageCallback2(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2);
    void imageCallback4(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2, const sensor_msgs::ImageConstPtr& img3, const sensor_msgs::ImageConstPtr& img4);
    
    void __processing(const std::vector<std::shared_ptr<cv::Mat>>& img_batch);
    void publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs);
    void publishBEVInfo();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber image_sub_;
    
    std::vector<message_filters::Subscriber<sensor_msgs::Image>> image_subs_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy4>> sync4_;
    
    ros::Publisher bev_info_pub_;
    std::vector<ros::Publisher> viz_result_pubs_;
    
    Model perception_model_;

    xt::xarray<double> Hs_; // (batch, 3, 3)
};

#endif