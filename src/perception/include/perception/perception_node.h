#ifndef __PERCEPTION_NODE_H__
#define __PERCEPTION_NODE_H__

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

#include "TAFv25.h"

#include <queue>
#include <memory>

#include <mutex>
#include <thread>
#include <condition_variable>

typedef std::shared_ptr<cv::Mat> MatPtr;

class PerceptionNode{
public:
    PerceptionNode();
    ~PerceptionNode();

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    
    std::string image_topic_name_;

    std::queue<MatPtr> buf_img_;
    std::mutex m_buf_;
    std::condition_variable cv_buf_;
    std::thread perception_thread;
    bool running_;

    TAFv25 perception_model_;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void processing();
};

#endif