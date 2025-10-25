#ifndef __PERCEPTION_NODE_H__
#define __PERCEPTION_NODE_H__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

#include "Model.h"

#include <queue>
#include <memory>

#include <mutex>
#include <thread>
#include <condition_variable>

typedef std::shared_ptr<cv::Mat> MatPtr;

class PerceptionNode{
public:
    PerceptionNode(
        const std::string& pkg_path,
        const std::string& image_topic_name,
        const int batch_size);
    ~PerceptionNode();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void processing();

    void publishBEVInfo();
    void publishVizResult();

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    
    bool running_;
    std::queue<MatPtr> buf_img_;
    std::mutex m_buf_;
    std::condition_variable cv_buf_;
    std::thread perception_thread;
    
    Model perception_model_;
};

#endif