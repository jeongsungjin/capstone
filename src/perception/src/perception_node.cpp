#include "perception/perception_node.h"

#include <string>

#include <cv_bridge/cv_bridge.h>

PerceptionNode::PerceptionNode(): nh_(), perception_model_(1920, 1080){
    ros::param::param<std::string>(
        "~image_topic", 
        image_topic_name_, 
        "/camera/image_raw"
    );
    
    image_sub_ = nh_.subscribe(image_topic_name_, 10, &PerceptionNode::imageCallback, this);
}

void PerceptionNode::imageCallback(const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image;
    if(!img.empty()){
        std::unique_lock<std::mutex> lock(m_buf_);
        buf_img_.push(std::make_shared<cv::Mat>(img));
    }

    cv_buf_.notify_one();
}

void PerceptionNode::processing(){
    ROS_INFO("Processing thread started.");
    while (running_){
        MatPtr img_ptr = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_buf_);
            cv_buf_.wait(lock, [this] { return !buf_img_.empty() || !running_; });
     
            if (!running_) break;
        
            img_ptr = buf_img_.front();
            buf_img_.pop();
        }

        if(img_ptr != nullptr) {
            auto model_input = perception_model_.preprocess(*img_ptr);
            auto model_output = perception_model_.inference(model_input);
            auto detections = perception_model_.postprocess(model_output);
            
            auto bev_info = perception_model_.toBEV(detections);
            perception_model_.visualizeDetections(*img_ptr, detections);
        }
    }
}
