#include "perception/model.h"
#include "perception/perception_node.h"
#include <rclcpp_components/register_node_macro.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>

using namespace std::chrono_literals;

PerceptionNode::PerceptionNode(const rclcpp::NodeOptions& options)
	: Node("perception_node", options), batch_size_(4)
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

	for(int i = 0; i < 4; i++){
		auto pub = this->create_publisher<sensor_msgs::msg::CompressedImage>(
			"viz_result/cam" + std::to_string(i+1) + "/compressed", 
			rclcpp::SensorDataQoS()
		);
		viz_result_pubs_.emplace_back(pub);
	}
	
	test_pub_ = this->create_publisher<std_msgs::msg::Int32>("test_int", 10);

	model_ = std::make_unique<Model>(pkg_path_, batch_size_);

	sub_a_ = std::make_unique<ImgSubscriber>(this, "/ipcam_1/image_raw/compressed");
	sub_b_ = std::make_unique<ImgSubscriber>(this, "/ipcam_2/image_raw/compressed");
	sub_c_ = std::make_unique<ImgSubscriber>(this, "/ipcam_3/image_raw/compressed");
	sub_d_ = std::make_unique<ImgSubscriber>(this, "/ipcam_4/image_raw/compressed");

	// increase queue size to buffer more messages and set a maximum allowed interval
	sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), *sub_a_, *sub_b_, *sub_c_, *sub_d_);
	// Wait at most 50 ms for matching messages — tune this (20-200ms) depending on jitter
	sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.05));
	sync_->registerCallback(std::bind(&PerceptionNode::syncCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

	RCLCPP_INFO(this->get_logger(), "PerceptionNode started; listening 4 image topics and feeding model (batch=%d)", batch_size_);

	// init sync rate monitor
	sync_count_.store(0);
	sync_start_ = std::chrono::steady_clock::now();
}

void PerceptionNode::syncCallback(const ImageMsg::ConstSharedPtr& a,
								  const ImageMsg::ConstSharedPtr& b,
								  const ImageMsg::ConstSharedPtr& c,
								  const ImageMsg::ConstSharedPtr& d)
{
	// increment sync counter and possibly log rate once per second
	sync_count_.fetch_add(1, std::memory_order_relaxed);
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - sync_start_).count();
	if (elapsed >= 1000) {
		uint64_t cnt = sync_count_.exchange(0);
		sync_start_ = now;
		double hz = static_cast<double>(cnt) / (static_cast<double>(elapsed) / 1000.0);
		RCLCPP_INFO(this->get_logger(), "[Perception] syncCallback called: %lu times in %.3f s -> %.2f Hz", cnt, static_cast<double>(elapsed)/1000.0, hz);
	}

	// compute header stamp spread among the 4 messages for diagnosis
	try {
		rclcpp::Time ta(a->header.stamp);
		rclcpp::Time tb(b->header.stamp);
		rclcpp::Time tc(c->header.stamp);
		rclcpp::Time td(d->header.stamp);
		auto na = static_cast<int64_t>(ta.nanoseconds());
		auto nb = static_cast<int64_t>(tb.nanoseconds());
		auto nc = static_cast<int64_t>(tc.nanoseconds());
		auto nd = static_cast<int64_t>(td.nanoseconds());
		int64_t tmin = std::min({na, nb, nc, nd});
		int64_t tmax = std::max({na, nb, nc, nd});
		double delta_ms = static_cast<double>(tmax - tmin) / 1e6;
		// warn if timestamps among the 4 images differ significantly
		const double WARN_THRESHOLD_MS = 20.0; // tuneable
		if (delta_ms > WARN_THRESHOLD_MS) {
			RCLCPP_WARN(this->get_logger(), "[Perception] header.stamp spread = %.2f ms (min=%ld max=%ld)", delta_ms, tmin, tmax);
		}
	} catch (...) {
		// ignore any stamp conversion issues
	}

	std::vector<std::shared_ptr<cv::Mat>> images;
	images.reserve(4);

	try {
		auto ca = cv_bridge::toCvCopy(a, "bgr8");
		auto cb = cv_bridge::toCvCopy(b, "bgr8");
		auto cc = cv_bridge::toCvCopy(c, "bgr8");
		auto cd = cv_bridge::toCvCopy(d, "bgr8");

		images.emplace_back(std::make_shared<cv::Mat>(ca->image));
		images.emplace_back(std::make_shared<cv::Mat>(cb->image));
		images.emplace_back(std::make_shared<cv::Mat>(cc->image));
		images.emplace_back(std::make_shared<cv::Mat>(cd->image));
	} catch (const cv_bridge::Exception& e) {
		RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
		return;
	}

	int ret = model_->preprocess(images);
	if(ret != 0){
		RCLCPP_WARN(this->get_logger(), "Model preprocess returned %d", ret);
		return;
	}

	model_->inference();
	model_->postprocess();

	test_pub_->publish(std_msgs::msg::Int32());

    // publishBEVInfo();
    publishVizResult(images);
}

void PerceptionNode::publishVizResult(const std::vector<std::shared_ptr<cv::Mat>>& imgs){
	const auto& detections = model_->getDetections();

	for (size_t b = 0; b < detections.size(); ++b) {
		cv_bridge::CvImage out_msg;
		out_msg.encoding = sensor_msgs::image_encodings::BGR8;
		imgs[b]->copyTo(out_msg.image);
		out_msg.header.stamp = this->get_clock()->now();

		for (size_t i = 0; i < detections[b].poly4s.size(); ++i) {
			std::vector<cv::Point> poly = {
				cv::Point(detections[b].poly4s[i](0, 0), detections[b].poly4s[i](0, 1)),
				cv::Point(detections[b].poly4s[i](1, 0), detections[b].poly4s[i](1, 1)),
				cv::Point(detections[b].poly4s[i](2, 0), detections[b].poly4s[i](2, 1)),
				cv::Point(detections[b].poly4s[i](3, 0), detections[b].poly4s[i](3, 1))
			};
			const std::vector<std::vector<cv::Point>> polys{poly};
			cv::polylines(out_msg.image, polys, true, cv::Scalar(0, 255, 0), 2);
		}

		auto msg = out_msg.toCompressedImageMsg();
		viz_result_pubs_[b]->publish(*msg);
	}
}

RCLCPP_COMPONENTS_REGISTER_NODE(PerceptionNode)
