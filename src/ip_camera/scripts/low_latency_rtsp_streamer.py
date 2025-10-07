#!/usr/bin/env python3
"""ROS node that ingests TP-Link VIGI C440I RTSP streams with sub-400 ms glass-to-glass latency.

The node uses a GStreamer pipeline tuned for minimum buffering, aggressive frame dropping,
optional hardware decode, and publishes only the freshest frame onto the requested topics.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

# Ensure GStreamer is initialised only once per process
Gst.init(None)


@dataclass
class CameraConfig:
    name: str
    ip: str
    username: str
    password: str
    stream: str
    topic: str
    frame_id: str
    transport: str = "udp"
    latency_ms: int = 80
    decoder: str = "auto"

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{self.username}:{self.password}@{self.ip}:554/{self.stream}"


def _is_decoder_available(factory_name: str) -> bool:
    registry = Gst.Registry.get()
    return registry.find_feature(factory_name, Gst.ElementFactory) is not None


class LowLatencyCamera(threading.Thread):
    """One RTSP ingest pipeline based on GStreamer appsink."""

    # Preferred decoders (fastest → slowest)
    DECODER_CANDIDATES = [
        "nvh264dec",
        "vaapih264dec",
        "v4l2h264dec",
        "avdec_h264",
    ]

    def __init__(self, config: CameraConfig):
        super().__init__(daemon=True)
        self.config = config
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_stamp = rospy.Time(0)
        self.frame_seq = 0
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[Gst.Element] = None
        self._bus: Optional[Gst.Bus] = None
        self._decoder_name = self._pick_decoder(config.decoder)
        self._backoff_seconds = 1.0
        self._max_backoff_seconds = 5.0
        
        # Stream candidates for fallback (requested -> stream2 -> stream1)
        candidates = [config.stream, "stream2", "stream1"]
        self._stream_candidates = []
        for s in candidates:
            if s not in self._stream_candidates:
                self._stream_candidates.append(s)
        self._stream_index = 0

    def _pick_decoder(self, requested: str) -> str:
        if requested != "auto":
            return requested
        for candidate in self.DECODER_CANDIDATES:
            if _is_decoder_available(candidate):
                return candidate
        return "avdec_h264"

    def stop(self):
        self._stop_event.set()

    

    def _build_pipeline_description(self) -> str:
        transport = "udp" if self.config.transport.lower() == "udp" else "tcp"
        stream_name = self._stream_candidates[self._stream_index]
        rtsp_url = f"rtsp://{self.config.username}:{self.config.password}@{self.config.ip}:554/{stream_name}"
        base = (
            f"rtspsrc name=src location={rtsp_url} latency={self.config.latency_ms} "
            f"protocols={transport} drop-on-late=false buffer-mode=none timeout=2000000 do-rtcp=true "
            f"ntp-sync=false")

        # Insert leaky queues to ensure old frames are discarded quickly
        queue_1 = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2"
        queue_2 = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2"
        queue_3 = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2"

        decoder_block = self._decoder_block()
        pipeline = (
            f"{base} ! rtpjitterbuffer mode=1 do-lost=true ! {queue_1} ! rtph264depay ! "
            f"h264parse config-interval=1 disable-passthrough=true ! {queue_2} ! {decoder_block} ! "
            f"videoconvert n-threads=1 ! video/x-raw,format=BGR ! {queue_3} ! "
            f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true qos=true"
        )
        return pipeline

    def _decoder_block(self) -> str:
        name = self._decoder_name
        if name == "avdec_h264":
            return "avdec_h264 max-threads=1 lowres=0"
        if name == "nvh264dec":
            # disable timing reordering when available
            return "nvh264dec disable-dpb=true drop-frame-interval=0"
        if name == "vaapih264dec":
            return "vaapih264dec low-latency=true"
        if name == "v4l2h264dec":
            return "v4l2h264dec capture-io-mode=dmabuf-import"
        return name

    def run(self):
        while not rospy.is_shutdown() and not self._stop_event.is_set():
            try:
                self._start_pipeline()
                self._mainloop()
                # Reset backoff after a successful session
                self._backoff_seconds = 1.0
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn(
                    "[%s] pipeline failure (%s). Reconnecting in %.1f s",
                    self.config.name,
                    exc,
                    self._backoff_seconds,
                )
                self._cleanup_pipeline()
                # If decodebin was requested, try forcing avdec_h264 on next attempt
                if self._decoder_name == "decodebin":
                    rospy.logwarn("[%s] forcing decoder fallback to avdec_h264", self.config.name)
                    self._decoder_name = "avdec_h264"
                # move to next stream candidate
                if len(self._stream_candidates) > 1:
                    self._stream_index = (self._stream_index + 1) % len(self._stream_candidates)
                    rospy.logwarn("[%s] switching to stream '%s'", self.config.name, self._stream_candidates[self._stream_index])
                if self._stop_event.wait(self._backoff_seconds):
                    break
                self._backoff_seconds = min(self._max_backoff_seconds, self._backoff_seconds * 1.7)
            else:
                # EOS reached without explicit stop: reconnect quickly
                if not self._stop_event.wait(self._backoff_seconds):
                    self._backoff_seconds = min(self._max_backoff_seconds, self._backoff_seconds * 1.3)
        self._cleanup_pipeline()

    def _start_pipeline(self):
        desc = self._build_pipeline_description()
        rospy.loginfo("[%s] launching pipeline: %s", self.config.name, desc)
        self._pipeline = Gst.parse_launch(desc)
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("appsink not found in pipeline")
        self._appsink.set_property("emit-signals", True)
        self._appsink.connect("new-sample", self._on_new_sample)
        self._bus = self._pipeline.get_bus()
        self._pipeline.set_state(Gst.State.PLAYING)
        rospy.loginfo("[%s] pipeline PLAYING", self.config.name)
        

    def _mainloop(self):
        assert self._pipeline is not None
        assert self._bus is not None
        start_seq = self.frame_seq
        start_time = rospy.Time.now().to_sec()
        frame_timeout = 5.0  # 5초 타임아웃
        while not rospy.is_shutdown() and not self._stop_event.is_set():
            message = self._bus.timed_pop_filtered(
                200 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if message is None:
                # no frames for 3s → switch stream
                if (self.frame_seq == start_seq) and ((rospy.Time.now().to_sec() - start_time) > 3.0):
                    raise RuntimeError("no frames within 3s; switching stream candidate")
                continue
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                raise RuntimeError(f"GStreamer error: {err} ({debug})")
            if message.type == Gst.MessageType.EOS:
                raise RuntimeError("received EOS")

    def _cleanup_pipeline(self):
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsink = None
        self._bus = None

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        caps = sample.get_caps()
        info = GstVideo.VideoInfo()
        if not info.from_caps(caps):
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            rowstride = info.stride[0]
            if rowstride == info.width * 3:
                frame_array = np.frombuffer(map_info.data, dtype=np.uint8, count=info.height * rowstride)
                np_frame = frame_array.reshape((info.height, info.width, 3)).copy()
            else:
                frame_array = np.frombuffer(map_info.data, dtype=np.uint8, count=info.height * rowstride)
                reshaped = frame_array.reshape((info.height, rowstride))
                cropped = reshaped[:, : info.width * 3]
                np_frame = cropped.reshape((info.height, info.width, 3)).copy()
        finally:
            buffer.unmap(map_info)

        stamp = rospy.Time.now()
        with self._frame_lock:
            self.latest_frame = np_frame
            self.latest_stamp = stamp
            self.frame_seq += 1
            if self.frame_seq % 30 == 0:
                rospy.loginfo("[%s] appsink received %d frames", self.config.name, self.frame_seq)
        return Gst.FlowReturn.OK

    def fetch_latest(self, last_seq: int) -> Optional[Tuple[np.ndarray, rospy.Time, int]]:
        with self._frame_lock:
            if self.latest_frame is None or self.frame_seq == last_seq:
                return None
            return self.latest_frame, self.latest_stamp, self.frame_seq


class MultiCameraStreamer:
    def __init__(self):
        rospy.init_node("vigi_low_latency_streamer", anonymous=False)
        self.bridge = CvBridge()
        self.publish_rate_hz = rospy.get_param("~publish_rate_hz", 90.0)
        self.cameras = self._load_cameras()
        self.publishers = {
            cam.config.name: rospy.Publisher(cam.config.topic, Image, queue_size=1, tcp_nodelay=True)
            for cam in self.cameras
        }
        self._camera_sequences: Dict[str, int] = {cam.config.name: -1 for cam in self.cameras}
        self._running = True
        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self._publisher_thread.start()
        # Start cameras in parallel to avoid connection conflicts
        import time
        
        def start_camera(cam):
            try:
                cam.start()
                rospy.loginfo("[%s] Camera started successfully", cam.config.name)
            except Exception as e:
                rospy.logerr("[%s] Failed to start camera: %s", cam.config.name, e)
        
        # Start all cameras in parallel threads
        threads = []
        for cam in self.cameras:
            thread = threading.Thread(target=start_camera, args=(cam,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for cameras to initialize
        time.sleep(3)
        
        # Check which cameras are actually working
        working_cameras = []
        for cam in self.cameras:
            if cam._pipeline is not None:
                working_cameras.append(cam)
                rospy.loginfo("[%s] Camera is working", cam.config.name)
            else:
                rospy.logwarn("[%s] Camera failed to start", cam.config.name)
        
        if not working_cameras:
            raise rospy.ROSInitException("No cameras started successfully")
        
        rospy.loginfo("Successfully started %d out of %d cameras", len(working_cameras), len(self.cameras))
        rospy.on_shutdown(self.shutdown)

    def _load_cameras(self) -> Tuple[LowLatencyCamera, ...]:
        default = [
            {
                "name": "camera_1",
                "ip": "192.168.0.60",
                "username": "admin",
                "password": "<PASSWORD>",
                "stream": "stream2",
                "topic": "/camera/camera_1/image_raw",
                "frame_id": "camera_1_link",
                "transport": "udp",
                "latency_ms": 80,
            },
            {
                "name": "camera_2",
                "ip": "192.168.0.195",
                "username": "admin",
                "password": "<PASSWORD>",
                "stream": "stream2",
                "topic": "/camera/camera_2/image_raw",
                "frame_id": "camera_2_link",
                "transport": "udp",
                "latency_ms": 80,
            },
        ]
        camera_dicts = rospy.get_param("~cameras", default)
        cameras = []
        for cfg in camera_dicts:
            try:
                latency_raw = cfg.get("latency_ms", 80)
                try:
                    latency_val = int(float(latency_raw))
                except (ValueError, TypeError):
                    raise rospy.ROSInitException(
                        f"Invalid latency_ms for camera '{cfg.get('name', '?')}': {latency_raw}"
                    )

                camera_cfg = CameraConfig(
                    name=cfg["name"],
                    ip=cfg["ip"],
                    username=cfg.get("username", "admin"),
                    password=cfg.get("password", "<PASSWORD>"),
                    stream=cfg.get("stream", "stream2"),
                    topic=cfg.get("topic", f"/camera/{cfg['name']}/image_raw"),
                    frame_id=cfg.get("frame_id", f"{cfg['name']}_link"),
                    transport=cfg.get("transport", "udp"),
                    latency_ms=latency_val,
                    decoder=cfg.get("decoder", "auto"),
                )
            except KeyError as exc:
                raise rospy.ROSInitException(f"Missing camera config key: {exc}") from exc
            cameras.append(LowLatencyCamera(camera_cfg))
        if not cameras:
            raise rospy.ROSInitException("At least one camera must be configured")
        rospy.loginfo("Loaded %d camera configurations", len(cameras))
        for cam in cameras:
            rospy.loginfo(
                "[%s] topic=%s frame_id=%s transport=%s latency_ms=%d decoder=%s",
                cam.config.name,
                cam.config.topic,
                cam.config.frame_id,
                cam.config.transport,
                cam.config.latency_ms,
                cam._decoder_name,  # access for logging only
            )
        return tuple(cameras)

    def _publisher_loop(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and self._running:
            for cam in self.cameras:
                # print("aassssssssssssssssssssssssssssssss")
                last_seq = self._camera_sequences[cam.config.name]
                fetched = cam.fetch_latest(last_seq)
                if fetched is None:
                    continue
                frame, stamp, seq = fetched
                try:
                    msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                except Exception as exc:  # pylint: disable=broad-except
                    rospy.logwarn("[%s] cv_bridge conversion failed: %s", cam.config.name, exc)
                    continue
                msg.header.stamp = stamp
                msg.header.frame_id = cam.config.frame_id
                self.publishers[cam.config.name].publish(msg)
                self._camera_sequences[cam.config.name] = seq
            rate.sleep()

    def shutdown(self):
        if not self._running:
            return
        self._running = False
        rospy.loginfo("Shutting down low-latency streamer...")
        for cam in self.cameras:
            cam.stop()
        for cam in self.cameras:
            cam.join(timeout=2.0)
        self._publisher_thread.join(timeout=2.0)
        rospy.loginfo("Shutdown complete")


def main():
    try:
        MultiCameraStreamer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
