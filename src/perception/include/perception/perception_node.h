#ifndef __PERCEPTION_NODE_H__
#define __PERCEPTION_NODE_H__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>

#include "Model.h"

#include <queue>
#include <vector>
#include <memory>

#include <mutex>
#include <thread>
#include <condition_variable>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef std::shared_ptr<cv::Mat> MatPtr;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;

class PerceptionNode{
public:
    PerceptionNode(
        const std::string& pkg_path,
        const std::string& image_topic_name,
        const int batch_size);
    ~PerceptionNode();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2);

    void processing();

    void publishBEVInfo();
    void publishVizResult(const std::vector<cv::Mat>& imgs);

private:
    ros::NodeHandle nh_;
    // For now we support 2 synced image topics (batch=2)
    message_filters::Subscriber<sensor_msgs::Image> image_sub1_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub2_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    ros::Publisher bev_info_pub_;
    std::vector<ros::Publisher> viz_result_pubs_;

    bool running_;
    std::queue<MatPtr> buf_img_;
    std::mutex m_buf_;
    std::condition_variable cv_buf_;
    std::thread perception_thread_;
    
    Model perception_model_;
};

#endif