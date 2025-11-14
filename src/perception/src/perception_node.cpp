#include "ffmpeg_streamer_node.h"
#include <chrono>

using namespace std::chrono_literals;

// 생성자
FfmpegStreamerNode::FfmpegStreamerNode(const rclcpp::NodeOptions & options)
  : Node("ffmpeg_streamer_node", options)
{
  // ROS 파라미터 선언 및 가져오기
  this->declare_parameter<std::string>("rtsp_url", "rtsp://default.url");
  this->declare_parameter<std::string>("frame_id", "camera_link");
  this->declare_parameter<int>("width", 1920);
  this->declare_parameter<int>("height", 1080);
  
  this->get_parameter("rtsp_url", rtsp_url_);
  this->get_parameter("frame_id", frame_id_);
  this->get_parameter("width", width_);
  this->get_parameter("height", height_);

  // 이미지 1장의 바이트 수 계산 (BGR8 가정)
  image_size_bytes_ = width_ * height_ * 3;

  RCLCPP_INFO(this->get_logger(), "Connecting to camera: %s", rtsp_url_.c_str());

  // ⭐️ (실제 코드 위치) ⭐️
  // 이 부분에서 FFmpeg (libavformat, libavcodec)을 초기화하고
  // rtsp_url_에 연결하는 코드를 작성해야 합니다.
  // ...

  // ROS 2 퍼블리셔 생성 (Intra-process 통신 활성화)
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
    "image_raw", 
    10, // QoS
    pub_options
  );

  // 30fps로 프레임을 발행하기 위한 타이머 생성
  timer_ = this->create_wall_timer(
    33ms, // 약 30fps
    std::bind(&FfmpegStreamerNode::publish_frame, this)
  );
}

// 소멸자
FfmpegStreamerNode::~FfmpegStreamerNode()
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up FFmpeg resources...");
  // ⭐️ (실제 코드 위치) ⭐️
  // 이 부분에서 avformat_close_input, avcodec_free_context 등을 호출하여
  // 모든 FFmpeg 리소스를 안전하게 해제해야 합니다.
}

// ⭐️ FFmpeg 디코딩 임시 함수 ⭐️
// 실제로는 FFmpeg의 av_read_frame, avcodec_send_packet, avcodec_receive_frame 등을
// 호출하여 디코딩된 프레임을 output_buffer에 직접 복사(또는 씌우기)합니다.
bool FfmpegStreamerNode::placeholder_ffmpeg_decode(uint8_t* output_buffer, size_t buffer_size)
{
  // (디코딩 성공했다고 가정)
  // 임시로 버퍼를 회색(128)으로 채웁니다.
  if (output_buffer) {
    memset(output_buffer, 128, buffer_size);
    return true;
  }
  return false;
}

// 타이머 콜백 (메인 루프)
void FfmpegStreamerNode::publish_frame()
{
  try {
    // 1. [빌리기] ROS 2 내부 메모리 풀에서 메시지를 "빌려옵니다".
    // 이 시점에는 데이터 복사가 전혀 없습니다.
    auto loaned_msg = image_publisher_->borrow_loaned_message();

    // 2. [참조 얻기] 빌려온 메시지 객체에 대한 참조를 얻습니다.
    sensor_msgs::msg::Image& msg = loaned_msg.get();

    // 3. [버퍼 할당] data 벡터의 크기를 재조정하여 메모리를 할당시킵니다.
    // (매번 할 필요는 없지만, 안전을 위해 예시에 포함)
    msg.data.resize(image_size_bytes_);

    // 4. ⭐️ [핵심] ⭐️
    // data 벡터가 할당한 내부 메모리 버퍼의 포인터를 가져옵니다.
    uint8_t* destination_buffer = msg.data.data();

    // 5. [데이터 씌우기] FFmpeg 디코더를 호출하여,
    //    ROS 2가 빌려준 메모리(destination_buffer)에 디코딩 결과를 "직접" 씁니다.
    bool decode_success = placeholder_ffmpeg_decode(destination_buffer, image_size_bytes_);

    if (decode_success) {
      // 6. [메타데이터 채우기]
      msg.header.stamp = this->get_clock()->now();
      msg.header.frame_id = frame_id_;
      msg.width = width_;
      msg.height = height_;
      msg.encoding = encoding_;
      msg.step = width_ * 3; // BGR8
      msg.is_bigendian = false;

      // 7. [발행] 데이터가 채워진 "빌린" 메시지를 발행합니다.
      //    소유권이 ROS 2 미들웨어로 넘어가며, 데이터 복사는 없습니다.
      image_publisher_->publish(std::move(loaned_msg));
    }
    // (decode_success가 false이면, loaned_msg는 아무것도 안 하고
    //  자동으로 소멸되며 메모리가 반환됩니다.)

  } catch (const rclcpp::exceptions::MessageLoanError & e) {
    RCLCPP_WARN(this->get_logger(), "Failed to loan message: %s", e.what());
  }
}