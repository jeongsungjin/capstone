#include "perception/perception_node.h"

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include <chrono>

PerceptionNode::PerceptionNode(
    const std::string& pkg_path, 
    const std::string& image_topic_name, 
    const int batch_size
): nh_(), perception_model_(pkg_path, batch_size)
{
    image_sub_ = nh_.subscribe(image_topic_name, 10, &PerceptionNode::imageCallback, this);

    running_ = true;
    perception_thread = std::thread(&PerceptionNode::processing, this);
}

void PerceptionNode::imageCallback(const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image;
    // if(!img.empty()){
    //     std::unique_lock<std::mutex> lock(m_buf_);
    //     buf_img_.push(std::make_shared<cv::Mat>(img));
    // }

    // cv_buf_.notify_one();

    int ret = perception_model_.preprocess(img);
    if (ret != 0) {
        running_ = false;
        std::cerr << "Preprocessing failed!" << std::endl;
        return;
    }

    perception_model_.inference();         
    perception_model_.postprocess();

    const auto& detections = perception_model_.getDetections();
    if(!detections.empty() && !detections[0].poly4s.empty()){   
        for(int i = 0; i < detections[0].poly4s.size(); i++){
            cv::polylines(img, 
                std::vector<std::vector<cv::Point>>{
                    {
                        cv::Point(detections[0].poly4s[i](0, 0), detections[0].poly4s[i](0, 1)),
                        cv::Point(detections[0].poly4s[i](2, 0), detections[0].poly4s[i](2, 1)),
                        cv::Point(detections[0].poly4s[i](4, 0), detections[0].poly4s[i](4, 1)),
                        cv::Point(detections[0].poly4s[i](6, 0), detections[0].poly4s[i](6, 1))
                    }
                }, 
                true, 
                cv::Scalar(0, 255, 0), 
                2
            );
        }
    }
        
    cv::imshow("Perception Result", img);
    cv::waitKey(1);
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

            perception_model_.inference();         
            perception_model_.postprocess();

            const auto& detections = perception_model_.getDetections();
            if(!detections.empty() && !detections[0].poly4s.empty()){   
                for(int i = 0; i < detections[0].poly4s.size(); i++){
                    cv::polylines(*img_ptr, 
                        std::vector<std::vector<cv::Point>>{
                            {
                                cv::Point(detections[0].poly4s[i](0, 0), detections[0].poly4s[i](0, 1)),
                                cv::Point(detections[0].poly4s[i](2, 0), detections[0].poly4s[i](2, 1)),
                                cv::Point(detections[0].poly4s[i](4, 0), detections[0].poly4s[i](4, 1)),
                                cv::Point(detections[0].poly4s[i](6, 0), detections[0].poly4s[i](6, 1))
                            }
                        }, 
                        true, 
                        cv::Scalar(0, 255, 0), 
                        2
                    );
                }
            }
                
            cv::imshow("Perception Result", *img_ptr);
            cv::waitKey(1);
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
