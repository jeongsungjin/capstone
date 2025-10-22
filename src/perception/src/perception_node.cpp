#include "perception/perception_node.h"

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include <chrono>

PerceptionNode::PerceptionNode(const std::string& pkg_path): nh_(), perception_model_(pkg_path){
    ros::param::param<std::string>(
        "~image_topic", 
        image_topic_name_, 
        "/camera/image_raw"
    );
    
    // std::cout << image_topic_name_ << std::endl;

    // image_sub_ = nh_.subscribe(image_topic_name_, 10, &PerceptionNode::imageCallback, this);

    // running_ = true;
    // perception_thread = std::thread(&PerceptionNode::processing, this);
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
            int ret = perception_model_.preprocess(*img_ptr);
            if (ret != 0) {
                running_ = false;
                std::cerr << "Preprocessing failed!" << std::endl;
                break;
            }

            // perception_model_.inference(model_input);
                      
            // auto detections = perception_model_.postprocess(model_output);
            
            // publishVizResult(*img_ptr, detections);
            // publishBEVInfo(*img_ptr, detections);
        }
    }
}

PerceptionNode::~PerceptionNode(){
    running_ = false;
    cv_buf_.notify_all();
    if (perception_thread.joinable()){
        perception_thread.join();
    }
}
