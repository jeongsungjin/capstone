#!/usr/bin/env python3
"""
최종 듀얼 카메라 스트리머
GStreamer 저지연 + 이미지 품질 보장
"""

import rospy
import cv2
import threading
import time
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FinalDualCamera:
    def __init__(self):
        rospy.init_node('final_dual_camera', anonymous=True)
        self.bridge = CvBridge()
        
        # 카메라 설정
        self.cameras = [
            {
                'name': 'camera_1',
                'url': 'rtsp://admin:zjsxmfhf@192.168.0.171:554/stream1',
                'topic': '/camera/camera_1/image_raw',
                'frame_id': 'camera_1_link'
            },
            {
                'name': 'camera_2', 
                'url': 'rtsp://admin:zjsxmfhf@192.168.0.195:554/stream1',
                'topic': '/camera/camera_2/image_raw',
                'frame_id': 'camera_2_link'
            }
        ]
        
        # 퍼블리셔 생성
        self.publishers = {}
        for cam in self.cameras:
            self.publishers[cam['name']] = rospy.Publisher(
                cam['topic'], Image, queue_size=1, tcp_nodelay=True
            )
        
        # 프레임 버퍼 (최신 1개만 유지)
        self.latest_frames = {}
        self.frame_locks = {}
        for cam in self.cameras:
            self.latest_frames[cam['name']] = None
            self.frame_locks[cam['name']] = threading.Lock()
        
        self.running = True
        
        rospy.loginfo("최종 듀얼 카메라 스트리머 시작")
        
    def create_gstreamer_pipeline(self, url):
        """GStreamer 파이프라인 생성 - 이미지 품질 보장"""
        pipeline = (
            f"rtspsrc location={url} "
            "latency=0 protocols=tcp drop-on-late=false buffer-mode=none "
            "timeout=2000000 do-rtcp=true ntp-sync=false ! "
            "rtpjitterbuffer mode=1 do-lost=true ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2 ! "
            "rtph264depay ! h264parse config-interval=1 disable-passthrough=true ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2 ! "
            "avdec_h264 max-threads=1 lowres=0 ! "
            "videoconvert n-threads=1 ! "
            "video/x-raw,format=BGR ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2 ! "
            "appsink emit-signals=true sync=false max-buffers=1 drop=true qos=true"
        )
        return pipeline
    
    def camera_capture_thread(self, camera_config):
        """카메라 캡처 전용 스레드 - GStreamer + 품질 보장"""
        name = camera_config['name']
        url = camera_config['url']
        
        rospy.loginfo(f"[{name}] GStreamer 연결 시도: {url}")
        
        # GStreamer 파이프라인 생성
        pipeline_str = self.create_gstreamer_pipeline(url)
        
        # OpenCV로 GStreamer 파이프라인 실행
        cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            rospy.logerr(f"[{name}] GStreamer 파이프라인 연결 실패")
            return
        
        rospy.loginfo(f"[{name}] GStreamer 연결 성공!")
        
        # 초기 버퍼 플러시
        for _ in range(5):
            cap.grab()
        
        frame_count = 0
        last_time = time.time()
        
        while self.running and not rospy.is_shutdown():
            # grab()으로 빠르게 프레임 가져오기
            if not cap.grab():
                rospy.logwarn(f"[{name}] grab 실패")
                continue
                
            # retrieve()로 실제 프레임 가져오기
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
            
            # 이미지 품질 검증
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                rospy.logwarn(f"[{name}] 이미지 크기 이상: {frame.shape}")
                continue
            
            current_time = time.time()
            
            # 프레임 카운트
            frame_count += 1
            if current_time - last_time >= 3.0:
                fps = frame_count / (current_time - last_time)
                rospy.loginfo(f"[{name}] 캡처 FPS: {fps:.1f}")
                frame_count = 0
                last_time = current_time
            
            # 최신 프레임만 저장
            with self.frame_locks[name]:
                self.latest_frames[name] = (frame.copy(), current_time)
                
        cap.release()
        rospy.loginfo(f"[{name}] 캡처 스레드 종료")
    
    def camera_publish_thread(self, camera_config):
        """카메라 발행 전용 스레드 - 극한 저지연 발행"""
        name = camera_config['name']
        topic = camera_config['topic']
        frame_id = camera_config['frame_id']
        
        rospy.loginfo(f"[{name}] 발행 스레드 시작")
        
        publish_count = 0
        last_time = time.time()
        last_frame_time = 0
        
        while self.running and not rospy.is_shutdown():
            # 최신 프레임 가져오기
            frame = None
            frame_time = None
            
            with self.frame_locks[name]:
                if self.latest_frames[name] is not None:
                    frame, frame_time = self.latest_frames[name]
            
            if frame is None or frame_time == last_frame_time:
                time.sleep(0.0001)  # 0.1ms 대기
                continue
            
            last_frame_time = frame_time
            
            # ROS 메시지로 변환 및 발행
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = frame_id
                self.publishers[name].publish(ros_image)
                
                publish_count += 1
                current_time = time.time()
                if current_time - last_time >= 3.0:
                    fps = publish_count / (current_time - last_time)
                    rospy.loginfo(f"[{name}] 발행 FPS: {fps:.1f}")
                    publish_count = 0
                    last_time = current_time
                    
            except Exception as e:
                rospy.logwarn(f"[{name}] 메시지 변환 실패: {e}")
                
        rospy.loginfo(f"[{name}] 발행 스레드 종료")
    
    def run(self):
        """메인 실행 함수"""
        # 각 카메라마다 캡처 스레드와 발행 스레드 생성
        threads = []
        
        for camera_config in self.cameras:
            # 캡처 스레드
            capture_thread = threading.Thread(
                target=self.camera_capture_thread,
                args=(camera_config,),
                daemon=True
            )
            capture_thread.start()
            threads.append(capture_thread)
            
            # 발행 스레드
            publish_thread = threading.Thread(
                target=self.camera_publish_thread,
                args=(camera_config,),
                daemon=True
            )
            publish_thread.start()
            threads.append(publish_thread)
            
            time.sleep(0.5)  # 카메라 간 연결 지연
        
        rospy.loginfo("모든 카메라 스레드 시작됨")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("종료 신호 수신")
        finally:
            self.running = False
            for thread in threads:
                thread.join(timeout=2)
            rospy.loginfo("모든 스레드 종료됨")

if __name__ == '__main__':
    try:
        streamer = FinalDualCamera()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
