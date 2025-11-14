#include "perception/perception_node.h"

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>

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

PerceptionNode::PerceptionNode(
    ros::NodeHandle nh, ros::NodeHandle pnh, const std::string& pkg_path, const int batch_size
): nh_(nh), pnh_(pnh), Hs_(xt::zeros<double>({batch_size, 3, 3})), perception_model_(pkg_path, batch_size), tracker_()
{
    if (batch_size != 2 && batch_size != 4) {
        ROS_WARN("PerceptionNode currently supports batch_size=2 for synchronized topics. Using first 2.");
    }

    // private ns 기반 발행
    bev_info_pub_ = pnh_.advertise<capstone_msgs::BEVInfo>("bev_info", 1);

    // 전역 ns 기반 parameter 가져오기 ( common.yaml )
    std::string topic_name_prefix;
    ros::param::param<std::string>(
        "/topic_name_prefix", topic_name_prefix, "/camera/image_raw"
    );

    ROS_INFO_STREAM("Image topic name prefix: " << topic_name_prefix);

    const int queue_size = 5;
    std::vector<ros::Publisher>(batch_size).swap(viz_result_pubs_);
    std::vector<message_filters::Subscriber<sensor_msgs::Image>>(batch_size).swap(image_subs_);
    for(int b = 0; b < batch_size; b++){
        // private ns 기반 발행 -> nodelet node 안에 정의된 param 에서 가져오는 것임!
        int cam_id = 0;
        pnh_.getParam("ipcam_" + std::to_string(b + 1), cam_id);
        
        const std::string& s_cam_id = std::to_string(cam_id);
        
        // private ns 기반 발행
        viz_result_pubs_[b] = pnh_.advertise<sensor_msgs::Image>("viz_result_" + s_cam_id, 1);
        
        // global ns 기반 parameter 가져오기 (개별 카메라에서 가져와야 하기 때문!)
        XmlRpc::XmlRpcValue H_param;
        nh_.getParam("/ipcam_" + s_cam_id + "/H", H_param);

        Hs_(b, 0, 0) = static_cast<double>(H_param[0][0]);
        Hs_(b, 0, 1) = static_cast<double>(H_param[0][1]);
        Hs_(b, 0, 2) = static_cast<double>(H_param[0][2]);

        Hs_(b, 1, 0) = static_cast<double>(H_param[1][0]);
        Hs_(b, 1, 1) = static_cast<double>(H_param[1][1]);
        Hs_(b, 1, 2) = static_cast<double>(H_param[1][2]);

        Hs_(b, 2, 0) = static_cast<double>(H_param[2][0]);
        Hs_(b, 2, 1) = static_cast<double>(H_param[2][1]);
        Hs_(b, 2, 2) = static_cast<double>(H_param[2][2]);
        
        // global ns 기반 구독
        ROS_INFO_STREAM("Subscribing to: " << topic_name_prefix + s_cam_id + "/image_raw");
        
        image_subs_[b].subscribe(nh_, topic_name_prefix + s_cam_id + "/image_raw", queue_size);
    }

    if(batch_size == 2){
        sync2_ = std::make_unique<message_filters::Synchronizer<SyncPolicy2>>(
            SyncPolicy2(queue_size), 
            image_subs_[0], 
            image_subs_[1]
        );

        using namespace boost::placeholders;
        sync2_->registerCallback(boost::bind(&PerceptionNode::imageCallback2, this, _1, _2));
    }
    
    else if(batch_size == 4){
        sync4_ = std::make_unique<message_filters::Synchronizer<SyncPolicy4>>(
            SyncPolicy4(queue_size), 
            image_subs_[0], 
            image_subs_[1],
            image_subs_[2],
            image_subs_[3]
        );

        using namespace boost::placeholders;
        sync4_->registerCallback(boost::bind(&PerceptionNode::imageCallback4, this, _1, _2, _3, _4));
    }
}

void PerceptionNode::imageCallback2(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img1_mat_ptr = std::make_shared<cv::Mat>(cv_ptr1->image);

    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img2_mat_ptr = std::make_shared<cv::Mat>(cv_ptr2->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img1_mat_ptr);
    img_batch.emplace_back(img2_mat_ptr);

    __processing(img_batch);
}

void PerceptionNode::imageCallback4(const sensor_msgs::ImageConstPtr& img1, const sensor_msgs::ImageConstPtr& img2, const sensor_msgs::ImageConstPtr& img3, const sensor_msgs::ImageConstPtr& img4){
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img1_mat_ptr = std::make_shared<cv::Mat>(cv_ptr1->image);

    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img2_mat_ptr = std::make_shared<cv::Mat>(cv_ptr2->image);

    cv_bridge::CvImagePtr cv_ptr3 = cv_bridge::toCvCopy(img3, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img3_mat_ptr = std::make_shared<cv::Mat>(cv_ptr3->image);

    cv_bridge::CvImagePtr cv_ptr4 = cv_bridge::toCvCopy(img4, sensor_msgs::image_encodings::BGR8);
    std::shared_ptr<cv::Mat> img4_mat_ptr = std::make_shared<cv::Mat>(cv_ptr4->image);

    std::vector<std::shared_ptr<cv::Mat>> img_batch;
    img_batch.emplace_back(img1_mat_ptr);
    img_batch.emplace_back(img2_mat_ptr);
    img_batch.emplace_back(img3_mat_ptr);
    img_batch.emplace_back(img4_mat_ptr);

    __processing(img_batch);
}

void PerceptionNode::__processing(const std::vector<std::shared_ptr<cv::Mat>>& img_batch){
    int ret = perception_model_.preprocess(img_batch);
    if (ret != 0) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return;
    }

    perception_model_.inference();
    perception_model_.postprocess();

    publishVizResult(img_batch);
    publishBEVInfo();
}

void PerceptionNode::publishBEVInfo(){
    const auto& detections = perception_model_.getDetections();

    // 1) Build detections for tracker
    std::vector<Detection> tracker_dets;
    tracker_dets.reserve(64);
    for(int b = 0; b < detections.size(); b++){
        auto H = xt::view(Hs_, b, xt::all(), xt::all());
        for(auto& tri_pts: detections[b].tri_ptss){
            auto center = xt::view(tri_pts, 0, xt::all());
            auto center3 = xt::concatenate(xt::xtuple(center, xt::xarray<double>({1.0})));
            auto bev_center = xt::linalg::dot(H, center3);
            bev_center /= bev_center(2);

            // Front mid and yaw
            auto front1 = xt::view(tri_pts, 1, xt::all());
            auto front2 = xt::view(tri_pts, 2, xt::all());
            auto front_center = (front1 + front2) / 2.0;
            auto front3 = xt::concatenate(xt::xtuple(front_center, xt::xarray<double>({1.0})));
            auto bev_front = xt::linalg::dot(H, front3);
            bev_front /= bev_front(2);
            double dx = static_cast<double>(bev_front(0) - bev_center(0));
            double dy = static_cast<double>(bev_front(1) - bev_center(1));
            double yaw_rad = std::atan2(dy, dx);
            double yaw_deg = yaw_rad * 180.0 / M_PI;

            // Approximate size from projected geometry
            // Width: distance between projected front corners
            auto f1p = xt::linalg::dot(H, xt::concatenate(xt::xtuple(front1, xt::xarray<double>({1.0}))));
            auto f2p = xt::linalg::dot(H, xt::concatenate(xt::xtuple(front2, xt::xarray<double>({1.0}))));
            f1p /= f1p(2); f2p /= f2p(2);
            double W = std::hypot(static_cast<double>(f2p(0) - f1p(0)), static_cast<double>(f2p(1) - f1p(1)));
            // Length: 2x distance center to front mid
            double L = 2.0 * std::hypot(dx, dy);

            Detection d;
            d.cls = 0; // class not provided; default 0
            d.x = static_cast<double>(bev_center(0));
            d.y = static_cast<double>(bev_center(1));
            d.L = L;
            d.W = W;
            d.yaw = yaw_deg; // tracker expects degrees
            tracker_dets.push_back(d);
        }
    }

    // 1.5) Cluster and fuse close detections across cameras before tracking
    double fuse_radius_m;               // cluster radius in BEV meters
    double yaw_fuse_deg_threshold;      // allow yaw within this difference inside a cluster
    std::string size_fuse_strategy;     // median | max | mean
    pnh_.param("fuse_radius_m", fuse_radius_m, 1.0);
    pnh_.param("yaw_fuse_deg_threshold", yaw_fuse_deg_threshold, 100.0);
    pnh_.param("size_fuse_strategy", size_fuse_strategy, std::string("mean"));

    auto ang_diff_deg = [](double a, double b){
        double d = std::fmod(a - b + 540.0, 360.0) - 180.0; // shortest signed diff
        return std::fabs(d);
    };

    std::vector<Detection> fused;
    fused.reserve(tracker_dets.size());
    const double r2 = fuse_radius_m * fuse_radius_m;
    std::vector<char> used(tracker_dets.size(), 0);
    for(size_t i=0; i<tracker_dets.size(); ++i){
        if(used[i]) continue;
        used[i] = 1;
        std::vector<size_t> cluster;
        cluster.push_back(i);
        // gather neighbors
        for(size_t j=i+1; j<tracker_dets.size(); ++j){
            if(used[j]) continue;
            double dx = tracker_dets[j].x - tracker_dets[i].x;
            double dy = tracker_dets[j].y - tracker_dets[i].y;
            if(dx*dx + dy*dy <= r2){
                // if(ang_diff_deg(tracker_dets[j].yaw, tracker_dets[i].yaw) <= yaw_fuse_deg_threshold){
                    used[j] = 1;
                    cluster.push_back(j);
                // }
            }
        }

        // fuse cluster
        double sumx=0.0, sumy=0.0, sum_sin=0.0, sum_cos=0.0;
        std::vector<double> Ls; Ls.reserve(cluster.size());
        std::vector<double> Ws; Ws.reserve(cluster.size());
        int cls = tracker_dets[cluster.front()].cls;
        for(size_t idx : cluster){
            const auto& d = tracker_dets[idx];
            sumx += d.x; sumy += d.y;
            double rad = d.yaw * M_PI / 180.0;
            sum_sin += std::sin(rad); sum_cos += std::cos(rad);
            Ls.push_back(d.L); Ws.push_back(d.W);
        }
        Detection out;
        out.cls = cls;
        out.x = sumx / cluster.size();
        out.y = sumy / cluster.size();
        double yaw_rad = std::atan2(sum_sin, sum_cos);
        out.yaw = yaw_rad * 180.0 / M_PI; // tracker expects degrees

        auto apply_strategy = [&](std::vector<double>& v){
            if(v.empty()) return 0.0;
            if(size_fuse_strategy == "max"){
                return *std::max_element(v.begin(), v.end());
            } else if(size_fuse_strategy == "mean"){
                double s = 0.0; for(double a: v) s += a; return s / v.size();
            } else { // median default
                std::sort(v.begin(), v.end());
                size_t n = v.size();
                if(n % 2 == 1) return v[n/2];
                else return 0.5*(v[n/2 - 1] + v[n/2]);
            }
        };
        out.L = apply_strategy(Ls);
        out.W = apply_strategy(Ws);

        fused.push_back(out);
    }
    tracker_dets.swap(fused);

    // 2) Update tracker and publish tracked outputs
    auto tracks = tracker_.update(tracker_dets);

    capstone_msgs::BEVInfo bev_info;
    bev_info.detCounts = static_cast<int32_t>(tracks.size());
    for(const auto& t : tracks){
        bev_info.ids.emplace_back(static_cast<int32_t>(t.track_id));
        bev_info.center_xs.emplace_back(static_cast<float>(t.x));
        bev_info.center_ys.emplace_back(static_cast<float>(t.y));
        // Convert back to radians for BEV message (previous code used radians)
        float yaw_rad = static_cast<float>(t.yaw * M_PI / 180.0);
        bev_info.yaws.emplace_back(yaw_rad);
    }

    bev_info_pub_.publish(bev_info);
}

void PerceptionNode::publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs){
    const auto& detections = perception_model_.getDetections();

    for(int b = 0; b < detections.size(); b++){
        cv_bridge::CvImage out_msg;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        imgs[b]->copyTo(out_msg.image);
        out_msg.header.stamp = ros::Time::now();
        
        for(int i = 0; i < detections[b].poly4s.size(); i++){
            cv::polylines(out_msg.image, 
                std::vector<std::vector<cv::Point>>{{
                    cv::Point(detections[b].poly4s[i](0, 0), detections[b].poly4s[i](0, 1)),
                    cv::Point(detections[b].poly4s[i](1, 0), detections[b].poly4s[i](1, 1)),
                    cv::Point(detections[b].poly4s[i](2, 0), detections[b].poly4s[i](2, 1)),
                    cv::Point(detections[b].poly4s[i](3, 0), detections[b].poly4s[i](3, 1))
                }}, 
                true, 
                cv::Scalar(0, 255, 0), 
                2
            );
        }
        
        viz_result_pubs_[b].publish(out_msg.toImageMsg());
    }
}

PerceptionNode::~PerceptionNode(){}
