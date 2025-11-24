#include "perception/model.h"
#include "perception/perception_node.h"
#include "ip_camera/rtsp_stream_manager.h"
#include <rclcpp_components/register_node_macro.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

PerceptionNode::PerceptionNode(const rclcpp::NodeOptions& options)
	: Node("perception_node", options), batch_size_(7)
{
	this->declare_parameter<std::string>("pkg_path", "");
	this->get_parameter("pkg_path", pkg_path_);
	if(pkg_path_.empty()){
		try{
			pkg_path_ = ament_index_cpp::get_package_share_directory("perception");
		} catch(...) {
			pkg_path_ = ".";
		}
	}

	auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
	detection_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
		"detection_info", 
		qos
	);

	model_ = std::make_unique<Model>(pkg_path_, batch_size_);

	// Load YAML configuration
	pkg_path_ = ament_index_cpp::get_package_share_directory("ip_camera");
	std::string yaml_path = pkg_path_ + "/config/ipcam.yaml";
	YAML::Node config = YAML::LoadFile(yaml_path);

	// Initialize RTSPStreamManager
	rtsp_manager_ = std::make_unique<RTSPStreamManager>(batch_size_);

	for (int i = 8; i < 8 + batch_size_; ++i) {
		std::string camera_key = "ipcam_" + std::to_string(i);
		if (config[camera_key]) {
			std::string ip = config[camera_key]["ros__parameters"]["ip"].as<std::string>();
			int port = config["/**"]["ros__parameters"]["port"].as<int>();
			std::string username = config["/**"]["ros__parameters"]["username"].as<std::string>();
			std::string password = config["/**"]["ros__parameters"]["password"].as<std::string>();

			std::string rtsp_url = "rtsp://" + username + ":" + password + "@" + ip + ":" + std::to_string(port) + "/stream1";
			rtsp_manager_->addStream(camera_key, rtsp_url, 10);
		} else {
			RCLCPP_WARN(this->get_logger(), "Camera configuration for %s not found in YAML", camera_key.c_str());
		}
	}

	RCLCPP_INFO(this->get_logger(), "PerceptionNode started; using RTSPStreamManager for %d streams.", batch_size_);

	// Start processing loop
	processing_thread_ = std::thread(&PerceptionNode::processStreams, this);
}

void PerceptionNode::processStreams() {
	rclcpp::WallRate rate(1000); // target processing cadence
	const auto per_stream_timeout = std::chrono::milliseconds(5); // per stream wait cap
	while (rclcpp::ok()) {
		// Attempt timed retrieval; skip incomplete batches without blocking indefinitely
		auto frames = rtsp_manager_->getAllFramesWithTimeout(per_stream_timeout);
		if (frames.size() != batch_size_) {
			// Not a full batch yet; short sleep to avoid busy spin
			rate.sleep();
			continue;
		}

		// for(auto frame: frames){
		// 	cv::imshow("Frame", frame);
		// }

		int ret = model_->preprocess(frames);
		if (ret != 0) {
			RCLCPP_WARN(this->get_logger(), "Model preprocess returned %d", ret);
			rate.sleep();
			continue;
		}
		model_->inference();
		model_->postprocess(detection_msg_);
		detection_pub_->publish(detection_msg_);
		// rate.sleep();

		// cv::waitKey(1);
	}
}

PerceptionNode::~PerceptionNode() {
	if (rtsp_manager_) {
		rtsp_manager_->stopAll();
	}
	if (processing_thread_.joinable()) {
		processing_thread_.join();
	}
}

RCLCPP_COMPONENTS_REGISTER_NODE(PerceptionNode)
