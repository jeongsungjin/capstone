#!/usr/bin/env python3
"""
VIGI C440I 전용 극한 저지연 스트리머
- 서브 스트림 우선 사용 (/stream2)
- 10 FPS, 320x240 해상도
- FFmpeg 저지연 옵션 적용
- 버퍼 크기 최소화
"""

import cv2
import rospy
import os
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VigiC440iUltraLowLatencyStreamer:
    def __init__(self):
        rospy.init_node('vigi_c440i_ultra_low_latency_streamer', anonymous=True)
        
        # FFmpeg 극한 저지연 환경변수 (최대한 공격적)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;256|max_delay;25000|fflags;nobuffer|flags;low_delay|probesize;8k|analyzeduration;0|sync;ext|threads;1'
        
        # OpenCV 최적화
        cv2.setNumThreads(1)
        
        # 카메라 설정 (VIGI C440I 최적화)
        self.cameras = [
            {
                'ip': '192.168.0.60',
                'username': 'admin',
                'password': 'zjsxmfhf',
                'topic': '/camera/camera_1/image_raw',
                'frame_id': 'camera_1_link'
            },
            {
                'ip': '192.168.0.195',
                'username': 'admin', 
                'password': 'zjsxmfhf',
                'topic': '/camera/camera_2/image_raw',
                'frame_id': 'camera_2_link'
            }
        ]
        
        # ROS 퍼블리셔 (극한 저지연)
        self.publishers = {}
        self.bridge = CvBridge()
        
        for i, camera in enumerate(self.cameras):
            self.publishers[i] = rospy.Publisher(
                camera['topic'],
                Image,
                queue_size=1,
                latch=False,
                tcp_nodelay=True
            )
        
        # 카메라 캡처 객체
        self.captures = {}
        self.running = True
        
        rospy.loginfo("🚀 VIGI C440I 극한 저지연 스트리머 시작!")
        
    def create_ultra_low_latency_url(self, camera):
        """극한 저지연 URL 생성 (서브 스트림 우선)"""
        # 서브 스트림 우선 (지연 최소화)
        urls = [
            f"rtsp://{camera['username']}:{camera['password']}@{camera['ip']}:554/stream2",  # 서브 스트림
            f"rtsp://{camera['username']}:{camera['password']}@{camera['ip']}:554/stream1"   # 메인 스트림
        ]
        return urls
    
    def setup_ultra_low_latency_camera(self, cap):
        """극한 저지연 카메라 설정"""
        # 버퍼 최소화
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 해상도 2560x1440 고정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        
        # FPS: 15 권장 (네트워크/디코딩 부하와 지연의 균형)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # H.264 고정 (Smart Coding 비활성화)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        # 추가 저지연 설정
        cap.set(cv2.CAP_PROP_FRAME_COUNT, 1)  # 프레임 카운트 최소화
        
    def connect_camera(self, camera_id, camera):
        """카메라 연결 (극한 저지연)"""
        urls = self.create_ultra_low_latency_url(camera)
        
        for url in urls:
            try:
                rospy.loginfo(f"🔗 카메라 {camera_id+1} 연결 시도: {url}")
                
                cap = cv2.VideoCapture(url)
                self.setup_ultra_low_latency_camera(cap)
                
                # 연결 테스트
                ret, frame = cap.read()
                if ret and frame is not None:
                    rospy.loginfo(f"✅ 카메라 {camera_id+1} 연결 성공!")
                    rospy.loginfo(f"   URL: {url}")
                    rospy.loginfo(f"   해상도: {frame.shape}")
                    return cap, url
                else:
                    cap.release()
                    
            except Exception as e:
                rospy.logwarn(f"❌ 카메라 {camera_id+1} 연결 실패: {str(e)}")
                if 'cap' in locals():
                    cap.release()
        
        return None, None
    
    def stream_camera(self, camera_id, camera):
        """카메라 스트리밍 (극한 저지연)"""
        cap, url = self.connect_camera(camera_id, camera)
        
        if cap is None:
            rospy.logerr(f"❌ 카메라 {camera_id+1} 연결 실패")
            return
        
        rospy.loginfo(f"📹 카메라 {camera_id+1} 스트리밍 시작 (극한 저지연)")
        
        frame_count = 0
        last_time = time.time()
        
        while self.running and not rospy.is_shutdown():
            try:
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # 프레임 리사이즈 제거 (원 해상도 유지)
                    
                    # ROS 메시지 변환
                    msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = camera['frame_id']
                    
                    # 퍼블리시 (지연 최소화)
                    self.publishers[camera_id].publish(msg)
                    
                    frame_count += 1
                    
                    # FPS 모니터링
                    if frame_count % 50 == 0:
                        current_time = time.time()
                        fps = 50 / (current_time - last_time)
                        rospy.loginfo(f"📊 카메라 {camera_id+1} FPS: {fps:.1f}")
                        last_time = current_time
                
                else:
                    rospy.logwarn(f"⚠️ 카메라 {camera_id+1} 프레임 읽기 실패")
                    time.sleep(0.1)
                    
            except Exception as e:
                rospy.logerr(f"❌ 카메라 {camera_id+1} 스트리밍 오류: {str(e)}")
                time.sleep(0.1)
        
        cap.release()
        rospy.loginfo(f"🛑 카메라 {camera_id+1} 스트리밍 종료")
    
    def run(self):
        """메인 실행"""
        import threading
        
        # 각 카메라를 별도 스레드에서 실행
        threads = []
        for i, camera in enumerate(self.cameras):
            thread = threading.Thread(target=self.stream_camera, args=(i, camera))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        rospy.loginfo("🎯 모든 카메라 스트리밍 시작!")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("🛑 사용자에 의해 종료")
        finally:
            self.running = False
            rospy.loginfo("🏁 VIGI C440I 극한 저지연 스트리머 종료")

if __name__ == '__main__':
    try:
        streamer = VigiC440iUltraLowLatencyStreamer()
        streamer.run()
    except rospy.ROSInterruptException:
        pass
