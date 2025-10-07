#!/usr/bin/env python3
"""
최적화된 듀얼 카메라 스트리머
웹뷰 수준의 저지연 + 안정적인 듀얼 카메라 지원
"""

import rospy
import cv2
import threading
import time
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OptimizedDualCamera:
    def __init__(self):
        rospy.init_node('optimized_dual_camera', anonymous=True)
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
        
        # 프레임 버퍼 (최신 프레임만 유지)
        self.frame_buffers = {}
        self.frame_locks = {}
        for cam in self.cameras:
            self.frame_buffers[cam['name']] = deque(maxlen=1)
            self.frame_locks[cam['name']] = threading.Lock()
        
        self.running = True
        
        rospy.loginfo("최적화된 듀얼 카메라 스트리머 시작")
        
    def camera_capture_thread(self, camera_config):
        """카메라 캡처 전용 스레드 - 최대한 빠르게 프레임 수집"""
        name = camera_config['name']
        url = camera_config['url']
        
        rospy.loginfo(f"[{name}] 연결 시도: {url}")
        
        # OpenCV 캡처 설정 - 저지연 최적화
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # 저지연 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FFmpeg 저지연 옵션 설정
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        # 연결 테스트
        ret, frame = cap.read()
        if not ret:
            rospy.logerr(f"[{name}] 연결 실패")
            return
            
        rospy.loginfo(f"[{name}] 연결 성공! 프레임 크기: {frame.shape}")
        
        frame_count = 0
        last_time = time.time()
        last_frame_time = time.time()
        
        while self.running and not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn(f"[{name}] 프레임 읽기 실패")
                continue
            
            current_time = time.time()
            
            # 프레임 간격 체크 (너무 빠르면 스킵)
            if current_time - last_frame_time < 0.033:  # 30 FPS 제한
                continue
            last_frame_time = current_time
            
            # 프레임 카운트 및 FPS 계산
            frame_count += 1
            if current_time - last_time >= 5.0:  # 5초마다 로그
                fps = frame_count / (current_time - last_time)
                rospy.loginfo(f"[{name}] 캡처 FPS: {fps:.1f}, 프레임: {frame_count}")
                frame_count = 0
                last_time = current_time
            
            # 최신 프레임만 버퍼에 저장 (오래된 프레임 덮어쓰기)
            with self.frame_locks[name]:
                self.frame_buffers[name].append((frame.copy(), current_time))
                
        cap.release()
        rospy.loginfo(f"[{name}] 캡처 스레드 종료")
    
    def camera_publish_thread(self, camera_config):
        """카메라 발행 전용 스레드 - 최대한 빠르게 ROS 메시지 발행"""
        name = camera_config['name']
        topic = camera_config['topic']
        frame_id = camera_config['frame_id']
        
        rospy.loginfo(f"[{name}] 발행 스레드 시작")
        
        publish_count = 0
        last_time = time.time()
        
        while self.running and not rospy.is_shutdown():
            # 최신 프레임 가져오기
            frame = None
            frame_time = None
            
            with self.frame_locks[name]:
                if self.frame_buffers[name]:
                    frame, frame_time = self.frame_buffers[name][-1]  # 최신 프레임만
            
            if frame is None:
                time.sleep(0.001)  # 1ms 대기
                continue
            
            # ROS 메시지로 변환 및 발행
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = frame_id
                self.publishers[name].publish(ros_image)
                
                publish_count += 1
                current_time = time.time()
                if current_time - last_time >= 5.0:  # 5초마다 로그
                    fps = publish_count / (current_time - last_time)
                    rospy.loginfo(f"[{name}] 발행 FPS: {fps:.1f}, 프레임: {publish_count}")
                    publish_count = 0
                    last_time = current_time
                    
            except Exception as e:
                rospy.logwarn(f"[{name}] 메시지 변환 실패: {e}")
            
            # CPU 사용률 조절 (너무 빠르면 대기)
            time.sleep(0.001)  # 1ms 대기
                
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
        streamer = OptimizedDualCamera()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
