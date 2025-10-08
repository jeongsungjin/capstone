#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import threading
import time
import os
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ffmpeg
import numpy as np

class IPCameraStreamer:
    def __init__(self):
        rospy.init_node('ip_camera_streamer', anonymous=True)
        
        # CV Bridge 초기화
        self.bridge = CvBridge()
        
        # 카메라 설정 (FFmpeg 파이프 기반)
        self.camera_configs = [
            {
                'ip': '192.168.0.171',
                'port': 554,
                'username': 'admin',
                'password': 'zjsxmfhf',
                'topic_name': '/camera/camera_1/image_raw',
                'frame_id': 'camera_1_link',
                'camera_id': 1,
                'transport': 'tcp',   # 'tcp' or 'udp'
                'width': 1280,
                'height': 720,
            },
            {
                'ip': '192.168.0.195',
                'port': 554,
                'username': 'admin',
                'password': 'zjsxmfhf',
                'topic_name': '/camera/camera_2/image_raw',
                'frame_id': 'camera_2_link',
                'camera_id': 2,
                'transport': 'tcp',
                'width': 1280,
                'height': 720,
            }
        ]
        
        # ROS 퍼블리셔 초기화 (극한 저지연을 위해 큐 크기 1)
        self.publishers = {}
        for config in self.camera_configs:
            self.publishers[config['camera_id']] = rospy.Publisher(
                config['topic_name'], 
                Image, 
                queue_size=1,
                latch=False,    # 래치 비활성화
                tcp_nodelay=True  # TCP 지연 최소화
            )
        
        # 카메라 캡처 객체들
        self.captures = {}
        # 최신 프레임 1장만 보관
        self.latest = {cfg['camera_id']: deque(maxlen=1) for cfg in self.camera_configs}
        self.running = True
        
        # 각 카메라에 대한 스레드 시작
        self.threads = []
        for config in self.camera_configs:
            thread = threading.Thread(target=self.camera_thread, args=(config,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        # 퍼블리셔 전용 스레드 (캡처와 분리)
        self.pub_thread = threading.Thread(target=self.publisher_loop)
        self.pub_thread.daemon = True
        self.pub_thread.start()
        
        rospy.loginfo("IP Camera Streamer 초기화 완료")
        rospy.loginfo(f"카메라 1: {self.camera_configs[0]['ip']} -> {self.camera_configs[0]['topic_name']}")
        rospy.loginfo(f"카메라 2: {self.camera_configs[1]['ip']} -> {self.camera_configs[1]['topic_name']}")
    
        # OpenCV 최적화
        cv2.setNumThreads(1)
    
    def create_stream_urls(self, config):
        """RTSP URL 목록 생성 (서브 스트림 우선)"""
        return [
            f"rtsp://{config['username']}:{config['password']}@{config['ip']}:{config['port']}/stream2",
            f"rtsp://{config['username']}:{config['password']}@{config['ip']}:{config['port']}/stream1",
        ]

    def spawn_ffmpeg(self, url: str, width: int, height: int, transport: str):
        """FFmpeg 파이프라인 생성 및 시작"""
        rtsp_transport = transport if transport in ('tcp', 'udp') else 'tcp'
        try:
            process = (
                ffmpeg
                .input(
                    url,
                    rtsp_transport=rtsp_transport,
                    fflags='nobuffer',
                    flags='low_delay',
                    probesize='16k',
                    analyzeduration='0'
                )
                .filter('scale', width, height)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='passthrough')
                .global_args('-loglevel', 'error', '-nostats')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            return process
        except ffmpeg.Error as e:
            rospy.logerr(f"FFmpeg error: {e.stderr.decode(errors='ignore') if e.stderr else e}")
            return None
    
    def camera_thread(self, config):
        """개별 카메라 스트림 처리 스레드"""
        camera_id = config['camera_id']
        stream_urls = self.create_stream_urls(config)
        
        rospy.loginfo(f"카메라 {camera_id} 연결 시도 시작")
        
        # 카메라 연결 시도 (FFmpeg 파이프)
        process = None
        successful_url = None
        width = int(config.get('width', 1280))
        height = int(config.get('height', 720))
        bytes_per_frame = width * height * 3
        
        # 각 스트림 URL을 시도 (FFmpeg 백엔드)
        for i, stream_url in enumerate(stream_urls):
            if not self.running:
                break
                
            rospy.loginfo(f"카메라 {camera_id} URL 시도 ({i+1}/{len(stream_urls)}): {stream_url}")
            
            try:
                process = self.spawn_ffmpeg(
                    stream_url,
                    width=width,
                    height=height,
                    transport=config.get('transport', 'tcp')
                )
                if process is None:
                    raise RuntimeError('ffmpeg spawn failed')

                # 연결 테스트: 한 프레임 읽기
                test_bytes = process.stdout.read(bytes_per_frame)
                if test_bytes and len(test_bytes) == bytes_per_frame:
                    frame = np.frombuffer(test_bytes, np.uint8).reshape((height, width, 3))
                    rospy.loginfo(f"✅ 카메라 {camera_id} 연결 성공!")
                    rospy.loginfo(f"   성공한 URL: {stream_url}")
                    rospy.loginfo(f"   해상도: {frame.shape[1]}x{frame.shape[0]}")
                    successful_url = stream_url
                    break
                else:
                    rospy.logwarn(f"❌ 카메라 {camera_id} URL 실패(프레임 없음): {stream_url}")
                    if process:
                        try:
                            process.stdout.close()
                            process.stderr.close()
                            process.wait(timeout=1)
                        except Exception:
                            pass
                    process = None

            except Exception as e:
                rospy.logerr(f"❌ 카메라 {camera_id} URL 오류 ({stream_url}): {str(e)}")
                if process:
                    try:
                        process.stdout.close()
                        process.stderr.close()
                        process.wait(timeout=1)
                    except Exception:
                        pass
                process = None
            
            time.sleep(0.5)  # 빠른 재시도
        
        if process is None:
            rospy.logerr(f"카메라 {camera_id} 연결 실패 - 모든 URL 시도 완료")
            return
        
        self.captures[camera_id] = process

        frame_count, last_time = 0, time.time()
        while self.running and not rospy.is_shutdown():
            try:
                # FFmpeg 파이프에서 프레임 읽기
                in_bytes = process.stdout.read(bytes_per_frame)
                if not in_bytes:
                    rospy.logwarn(f"카메라 {camera_id} 스트림 종료/중단 감지, 재연결 시도")
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
                dq = self.latest[camera_id]
                dq.clear()
                dq.append(frame)

                frame_count += 1
                now = time.time()
                if now - last_time >= 5.0:
                    rospy.loginfo(f"카메라 {camera_id} capture fps≈{frame_count/(now-last_time):.2f}")
                    frame_count, last_time = 0, now
            except Exception as e:
                rospy.logerr(f"카메라 {camera_id} 스트리밍 오류: {str(e)}")
                time.sleep(0.05)
        
        # 정리
        if process:
            try:
                process.stdout.close()
                process.stderr.close()
                process.wait(timeout=1)
            except Exception:
                pass
        rospy.loginfo(f"카메라 {camera_id} 스트리밍 종료")

    def publisher_loop(self):
        rate = rospy.Rate(60)
        while self.running and not rospy.is_shutdown():
            stamp = rospy.Time.now()
            for cfg in self.camera_configs:
                cam_id = cfg['camera_id']
                if self.latest[cam_id]:
                    frame = self.latest[cam_id][-1]
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                        ros_image.header.stamp = stamp
                        ros_image.header.frame_id = cfg['frame_id']
                        self.publishers[cam_id].publish(ros_image)
                    except Exception as e:
                        rospy.logwarn(f"카메라 {cam_id} 퍼블리시 오류: {str(e)}")
            rate.sleep()
    
    def run(self):
        """메인 실행 함수"""
        try:
            rospy.loginfo("IP Camera Streamer 시작")
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("사용자에 의해 종료됨")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        rospy.loginfo("리소스 정리 중...")
        self.running = False
        
        # 모든 FFmpeg 프로세스 종료
        for proc in self.captures.values():
            if not proc:
                continue
            try:
                proc.stdout.close()
                proc.stderr.close()
                proc.wait(timeout=1)
            except Exception:
                pass
        
        # 스레드 종료 대기
        for thread in self.threads:
            thread.join(timeout=1)
        try:
            self.pub_thread.join(timeout=1)
        except Exception:
            pass
        
        rospy.loginfo("정리 완료")

def main():
    try:
        streamer = IPCameraStreamer()
        streamer.run()
    except Exception as e:
        rospy.logerr(f"IP Camera Streamer 오류: {str(e)}")

if __name__ == '__main__':
    main()
