#!/usr/bin/env python3
"""
극한 저지연 듀얼 카메라 스트리머
웹뷰 수준의 실시간 성능 + 안정적인 듀얼 카메라
"""

import rospy
import cv2
import threading
import time
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class UltraLowLatencyDualCamera:
    def __init__(self):
        rospy.init_node('ultra_low_latency_dual_camera', anonymous=True)
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
        
        rospy.loginfo("극한 저지연 듀얼 카메라 스트리머 시작")
        
    def camera_capture_thread(self, camera_config):
        """카메라 캡처 전용 스레드 - 극한 저지연 최적화"""
        name = camera_config['name']
        url = camera_config['url']
        
        rospy.loginfo(f"[{name}] 연결 시도: {url}")
        
        # OpenCV 캡처 설정 - 극한 저지연
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # 극한 저지연 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FFmpeg 저지연 옵션
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        # 연결 테스트
        ret, frame = cap.read()
        if not ret:
            rospy.logerr(f"[{name}] 연결 실패")
            return
            
        rospy.loginfo(f"[{name}] 연결 성공! 프레임 크기: {frame.shape}")
        
        # 초기 버퍼 플러시 (오래된 프레임 제거)
        for _ in range(10):
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
            if not ret:
                continue
            
            current_time = time.time()
            
            # 프레임 카운트
            frame_count += 1
            if current_time - last_time >= 3.0:  # 3초마다 로그
                fps = frame_count / (current_time - last_time)
                rospy.loginfo(f"[{name}] 캡처 FPS: {fps:.1f}")
                frame_count = 0
                last_time = current_time
            
            # 최신 프레임만 저장 (오래된 프레임 덮어쓰기)
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
                if current_time - last_time >= 3.0:  # 3초마다 로그
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
            
            time.sleep(0.3)  # 카메라 간 연결 지연
        
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
        streamer = UltraLowLatencyDualCamera()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
