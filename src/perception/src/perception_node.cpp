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
    
    std::cout << image_topic_name_ << std::endl;

    image_sub_ = nh_.subscribe(image_topic_name_, 10, &PerceptionNode::imageCallback, this);

    running_ = true;
    perception_thread = std::thread(&PerceptionNode::processing, this);
}

void PerceptionNode::imageCallback(const sensor_msgs::ImageConstPtr& msg){
    // ROS_INFO("I got message");

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image;
    if(!img.empty()){
        std::unique_lock<std::mutex> lock(m_buf_);
        buf_img_.push(std::make_shared<cv::Mat>(img));
    }

    cv_buf_.notify_one();
}

void PerceptionNode::processing(){
    // ROS_INFO("Processing thread started.");
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
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            auto model_input = perception_model_.preprocess(*img_ptr);
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Preprocess time: " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() 
                      << " ms" << std::endl;
                      
            std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
            auto model_output = perception_model_.inference(model_input);
            std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
            std::cout << "inference time: " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() 
                      << " ms" << std::endl;
                      
            std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
            auto detections = perception_model_.postprocess(model_output);
            std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
            std::cout << "poseprocess time: " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() 
                      << " ms" << std::endl;
            
            // auto bev_info = perception_model_.toBEV(detections);
            perception_model_.visualizeDetections(*img_ptr, detections);
            cv::imshow("Perception Detections", *img_ptr);
            cv::waitKey(1);
        }
    }
    // ROS_INFO("Processing thread started.");

}

PerceptionNode::~PerceptionNode(){
    running_ = false;
    cv_buf_.notify_all();
    if (perception_thread.joinable()){
        perception_thread.join();
    }
}
