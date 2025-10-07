#!/usr/bin/env python3
"""
GStreamer-based RTSP → ROS1 Image publisher with diagnostics and switchable profiles.

Problem context (root causes in practice):
- OpenCV/FFmpeg backends introduce unavoidable internal buffering and reordering, causing ~1s latency vs web view.
- GStreamer ultra-low-latency pipelines can suffer from block corruption or ghosting if the network or camera encoding settings are not ideal (B-frames, large GOP, heavy ISP noise reduction), or if the decoder synchronization drifts.

Approach in this node:
- Provide two selectable profiles via ~profile: "ultra" (lowest-latency) and "stable" (more tolerant, minimizes corruption) without changing camera settings.
- Expose key knobs (transport, latency_ms, decoder, parse options) as ROS params.
- Keep the latest frame only (queue_size=1, appsink drop) to avoid stale frames.
- Optional watchdog (off by default) that can request reconnects if frames stop updating.

Camera-side recommendations (most impactful for ghosting/latency):
- Disable B-frames, set GOP=FPS (e.g., 15 fps → 15), disable Smart Coding/SVC.
- Reduce 3D DNR to Low or Off; fix shutter speed near 1/60–1/120s to reduce motion smear.
- After changing camera settings, reboot cameras to ensure RTSP picks updated parameters.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import threading
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # type: ignore

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
    transport: str = "udp"            # "udp" or "tcp"
    latency_ms: int = 80               # rtspsrc/jitter LATENCY
    decoder: str = "auto"             # auto|avdec_h264|nvh264dec|vaapih264dec|v4l2h264dec

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{self.username}:{self.password}@{self.ip}:554/{self.stream}"


def _is_decoder_available(factory_name: str) -> bool:
    registry = Gst.Registry.get()
    return registry.find_feature(factory_name, Gst.ElementFactory) is not None


class RtspIngest(threading.Thread):
    DECODER_CANDIDATES = ["nvh264dec", "vaapih264dec", "v4l2h264dec", "avdec_h264"]

    def __init__(self, cfg: CameraConfig, profile: str, parse_alignment: bool, parse_config_interval: int,
                 buffer_mode: str, drop_on_late_src: bool, jitter_drop_on_late: bool, jitter_latency_ms: Optional[int]):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.profile = profile
        self.parse_alignment = parse_alignment
        self.parse_config_interval = parse_config_interval
        self.buffer_mode = buffer_mode
        self.drop_on_late_src = drop_on_late_src
        self.jitter_drop_on_late = jitter_drop_on_late
        self.jitter_latency_ms = jitter_latency_ms

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_stamp = rospy.Time(0)
        self.frame_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[Gst.Element] = None
        self._bus: Optional[Gst.Bus] = None
        self._decoder_name = self._pick_decoder(cfg.decoder)

    def _pick_decoder(self, requested: str) -> str:
        if requested != "auto":
            return requested
        for name in self.DECODER_CANDIDATES:
            if _is_decoder_available(name):
                return name
        return "avdec_h264"

    def _supports_h264parse_alignment(self) -> bool:
        try:
            element = Gst.ElementFactory.make("h264parse", None)
            if element is None:
                return False
            try:
                for prop in element.list_properties():  # type: ignore[attr-defined]
                    if getattr(prop, "name", "") == "alignment":
                        return True
            except Exception:
                pass
            try:
                return hasattr(element.props, "alignment")
            except Exception:
                return False
        except Exception:
            return False

    def _decoder_block(self) -> str:
        name = self._decoder_name
        if name == "avdec_h264":
            return "avdec_h264 max-threads=1 lowres=0"
        if name == "nvh264dec":
            return "nvh264dec disable-dpb=true drop-frame-interval=0"
        if name == "vaapih264dec":
            return "vaapih264dec low-latency=true"
        if name == "v4l2h264dec":
            return "v4l2h264dec capture-io-mode=dmabuf-import"
        return name

    def _build_pipeline(self) -> str:
        transport = "udp" if self.cfg.transport.lower() == "udp" else "tcp"
        parse_align_opt = "alignment=au " if (self.parse_alignment and self._supports_h264parse_alignment()) else ""
        jitter_lat = self.jitter_latency_ms if self.jitter_latency_ms is not None else self.cfg.latency_ms

        if self.profile == "stable":
            # Tolerant path: allow late drops, auto buffer mode, reinforce SPS/PPS
            base = (
                f"rtspsrc name=src location={self.cfg.rtsp_url} latency={self.cfg.latency_ms} "
                f"protocols={transport} drop-on-late=true buffer-mode={self.buffer_mode} timeout=5000000 do-rtcp=true ntp-sync=false"
            )
            jitter = f"rtpjitterbuffer latency={jitter_lat} mode=1 do-lost=true drop-on-late={str(self.jitter_drop_on_late).lower()}"
            h264parse_block = f"h264parse {parse_align_opt}config-interval=-1 disable-passthrough=true"
        else:
            # Ultra-low-latency default
            base = (
                f"rtspsrc name=src location={self.cfg.rtsp_url} latency={self.cfg.latency_ms} "
                f"protocols={transport} drop-on-late=false buffer-mode={self.buffer_mode} timeout=2000000 do-rtcp=true ntp-sync=false"
            )
            jitter = "rtpjitterbuffer mode=1 do-lost=true"
            h264parse_block = f"h264parse {parse_align_opt}config-interval={self.parse_config_interval} disable-passthrough=true"

        q = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=2"
        decoder = self._decoder_block()
        pipe = (
            f"{base} ! {jitter} ! {q} ! rtph264depay ! {h264parse_block} ! {q} ! {decoder} ! "
            f"videoconvert n-threads=1 ! video/x-raw,format=BGR ! {q} ! "
            f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true qos=true"
        )
        return pipe

    def run(self):
        backoff = 1.0
        max_backoff = 5.0
        while not rospy.is_shutdown() and not self._stop.is_set():
            try:
                desc = self._build_pipeline()
                rospy.loginfo("[%s] launching pipeline: %s", self.cfg.name, desc)
                self._pipeline = Gst.parse_launch(desc)
                self._appsink = self._pipeline.get_by_name("sink")
                if self._appsink is None:
                    raise RuntimeError("appsink not found")
                self._appsink.set_property("emit-signals", True)
                self._appsink.connect("new-sample", self._on_new_sample)
                self._bus = self._pipeline.get_bus()
                self._pipeline.set_state(Gst.State.PLAYING)
                rospy.loginfo("[%s] pipeline PLAYING", self.cfg.name)
                # mainloop
                start_seq = self.frame_seq
                start_time = time.time()
                while not rospy.is_shutdown() and not self._stop.is_set():
                    msg = self._bus.timed_pop_filtered(200 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                    if msg is None:
                        if (self.frame_seq == start_seq) and ((time.time() - start_time) > 3.0):
                            raise RuntimeError("no frames within 3s")
                        continue
                    if msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        raise RuntimeError(f"GStreamer error: {err} ({dbg})")
                    if msg.type == Gst.MessageType.EOS:
                        raise RuntimeError("received EOS")
                backoff = 1.0
            except Exception as e:  # noqa: BLE001
                rospy.logwarn("[%s] pipeline failure (%s). Reconnecting in %.1fs", self.cfg.name, e, backoff)
                self._cleanup()
                if self._stop.wait(backoff):
                    break
                backoff = min(max_backoff, backoff * 1.7)
            else:
                self._cleanup()

    def stop(self):
        self._stop.set()

    def _cleanup(self):
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsink = None
        self._bus = None

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:  # type: ignore[valid-type]
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        caps = sample.get_caps()
        info = GstVideo.VideoInfo()
        if not info.from_caps(caps):
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            rowstride = info.stride[0]
            if rowstride == info.width * 3:
                arr = np.frombuffer(map_info.data, dtype=np.uint8, count=info.height * rowstride)
                img = arr.reshape((info.height, info.width, 3)).copy()
            else:
                arr = np.frombuffer(map_info.data, dtype=np.uint8, count=info.height * rowstride)
                reshaped = arr.reshape((info.height, rowstride))
                cropped = reshaped[:, : info.width * 3]
                img = cropped.reshape((info.height, info.width, 3)).copy()
        finally:
            buffer.unmap(map_info)

        with self._lock:
            self.latest_frame = img
            self.latest_stamp = rospy.Time.now()
            self.frame_seq += 1
        return Gst.FlowReturn.OK

    def fetch_latest(self, last_seq: int) -> Optional[Tuple[np.ndarray, rospy.Time, int]]:
        with self._lock:
            if self.latest_frame is None or self.frame_seq == last_seq:
                return None
            return self.latest_frame, self.latest_stamp, self.frame_seq


class GStreamerRtspRosStreamer:
    def __init__(self):
        rospy.init_node("gst_rtsp_ros_streamer", anonymous=False)
        self.bridge = CvBridge()

        # Global knobs
        self.profile = str(rospy.get_param("~profile", "ultra")).lower()  # ultra|stable
        self.buffer_mode = str(rospy.get_param("~buffer_mode", "none"))   # none|auto
        self.drop_on_late_src = bool(rospy.get_param("~drop_on_late_src", False))
        self.jitter_drop_on_late = bool(rospy.get_param("~jitter_drop_on_late", False))
        self.jitter_latency_ms = rospy.get_param("~jitter_latency_ms", None)
        self.parse_alignment = bool(rospy.get_param("~parse_alignment", True))
        self.parse_config_interval = int(rospy.get_param("~parse_config_interval", 1))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 120.0))
        self.use_watchdog = bool(rospy.get_param("~use_watchdog", False))
        self.stale_timeout_sec = float(rospy.get_param("~stale_timeout_sec", 2.0))

        self.cameras = self._load_cameras()
        self.publishers = {cam.cfg.name: rospy.Publisher(cam.cfg.topic, Image, queue_size=1, tcp_nodelay=True)
                           for cam in self.cameras}
        self._last_seq: Dict[str, int] = {cam.cfg.name: -1 for cam in self.cameras}
        self._last_update: Dict[str, float] = {cam.cfg.name: time.time() for cam in self.cameras}

        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self._publisher_thread.start()
        rospy.on_shutdown(self.shutdown)

    def _load_cameras(self):
        default = [
            {"name": "camera_1", "ip": "192.168.0.171", "username": "admin", "password": "<PASSWORD>",
             "stream": "stream2", "topic": "/camera/camera_1/image_raw", "frame_id": "camera_1_link", "transport": "udp", "latency_ms": 80, "decoder": "auto"},
            {"name": "camera_2", "ip": "192.168.0.195", "username": "admin", "password": "<PASSWORD>",
             "stream": "stream2", "topic": "/camera/camera_2/image_raw", "frame_id": "camera_2_link", "transport": "udp", "latency_ms": 80, "decoder": "auto"},
        ]
        cam_dicts = rospy.get_param("~cameras", default)
        cameras = []
        for cfg in cam_dicts:
            cam_cfg = CameraConfig(
                name=cfg["name"], ip=cfg["ip"], username=cfg.get("username", "admin"),
                password=cfg.get("password", "<PASSWORD>"), stream=cfg.get("stream", "stream2"),
                topic=cfg.get("topic", f"/camera/{cfg['name']}/image_raw"), frame_id=cfg.get("frame_id", f"{cfg['name']}_link"),
                transport=cfg.get("transport", "udp"), latency_ms=int(cfg.get("latency_ms", 80)), decoder=cfg.get("decoder", "auto")
            )
            cameras.append(RtspIngest(
                cam_cfg, self.profile, self.parse_alignment, self.parse_config_interval,
                self.buffer_mode, self.drop_on_late_src, self.jitter_drop_on_late, self.jitter_latency_ms
            ))
        if not cameras:
            raise rospy.ROSInitException("At least one camera must be configured")
        for cam in cameras:
            cam.start()
        rospy.loginfo("Started %d camera pipelines (profile=%s, buffer_mode=%s)", len(cameras), self.profile, self.buffer_mode)
        return cameras

    def _publisher_loop(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            now = time.time()
            for cam in self.cameras:
                got = cam.fetch_latest(self._last_seq[cam.cfg.name])
                if got is None:
                    if self.use_watchdog and (now - self._last_update[cam.cfg.name]) > self.stale_timeout_sec:
                        rospy.logwarn("[%s] no frame for %.2fs (watchdog)", cam.cfg.name, now - self._last_update[cam.cfg.name])
                    continue
                frame, stamp, seq = got
                try:
                    msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                except Exception as exc:  # noqa: BLE001
                    rospy.logwarn("[%s] cv_bridge conversion failed: %s", cam.cfg.name, exc)
                    continue
                msg.header.stamp = stamp
                msg.header.frame_id = cam.cfg.frame_id
                self.publishers[cam.cfg.name].publish(msg)
                self._last_seq[cam.cfg.name] = seq
                self._last_update[cam.cfg.name] = now
            rate.sleep()

    def shutdown(self):
        for cam in self.cameras:
            cam.stop()
        for cam in self.cameras:
            cam.join(timeout=2.0)


def main():
    try:
        GStreamerRtspRosStreamer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


