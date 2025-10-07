#!/usr/bin/env python3
"""
간단한 듀얼 카메라 스트리머
OpenCV를 사용하여 두 카메라를 동시에 스트리밍
"""

import rospy
import cv2
import threading
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class SimpleDualCamera:
    def __init__(self):
        rospy.init_node('simple_dual_camera', anonymous=True)
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
        
        # 카메라 캡처 객체들
        self.captures = {}
        self.running = True
        
        rospy.loginfo("Simple Dual Camera 스트리머 시작")
        
    def camera_thread(self, camera_config):
        """개별 카메라 스레드"""
        name = camera_config['name']
        url = camera_config['url']
        topic = camera_config['topic']
        frame_id = camera_config['frame_id']
        
        rospy.loginfo(f"[{name}] 연결 시도: {url}")
        
        # OpenCV 캡처 설정
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 연결 테스트
        ret, frame = cap.read()
        if not ret:
            rospy.logerr(f"[{name}] 연결 실패")
            return
            
        rospy.loginfo(f"[{name}] 연결 성공! 프레임 크기: {frame.shape}")
        
        frame_count = 0
        last_time = time.time()
        
        while self.running and not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn(f"[{name}] 프레임 읽기 실패")
                continue
                
            # 프레임 카운트 및 FPS 계산
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 5.0:  # 5초마다 로그
                fps = frame_count / (current_time - last_time)
                rospy.loginfo(f"[{name}] FPS: {fps:.1f}, 프레임: {frame_count}")
                frame_count = 0
                last_time = current_time
            
            # ROS 메시지로 변환
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = frame_id
                self.publishers[name].publish(ros_image)
            except Exception as e:
                rospy.logwarn(f"[{name}] 메시지 변환 실패: {e}")
                
        cap.release()
        rospy.loginfo(f"[{name}] 스트리밍 종료")
    
    def run(self):
        """메인 실행 함수"""
        # 각 카메라를 별도 스레드에서 실행
        threads = []
        for camera_config in self.cameras:
            thread = threading.Thread(
                target=self.camera_thread, 
                args=(camera_config,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            time.sleep(1)  # 카메라 간 연결 지연
        
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
        streamer = SimpleDualCamera()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
