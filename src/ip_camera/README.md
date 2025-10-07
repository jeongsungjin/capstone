# IP Camera ROS1 Package

Low-latency ROS1 streaming for two TP-Link VIGI C440I cameras. The new
`low_latency_rtsp_streamer.py` node uses a GStreamer pipeline tuned for
sub-400 ms glass-to-glass latency while continuously discarding stale frames.

## 1. Dependencies

```bash
sudo apt-get install \
  ros-noetic-cv-bridge ros-noetic-sensor-msgs ros-noetic-image-transport \
  python3-gi gir1.2-gst-plugins-base-1.0 gir1.2-gst-plugins-bad-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Optional hardware decoders
sudo apt-get install gstreamer1.0-vaapi   # Intel iGPU
sudo apt-get install gstreamer1.0-nvvideocodec-plugins  # NVIDIA GPU
```

Python dependencies are satisfied by ROS desktop installations; no OpenCV
VideoCapture is required for the low-latency node.

## 2. Build

```bash
cd /home/ctrl/capstone_2025
catkin build ip_camera
source devel/setup.bash
```

## 3. Run the Low-Latency Streamer

```bash
roslaunch ip_camera vigi_low_latency.launch \
  camera_password:=<PASSWORD> \
  transport:=udp \
  latency_ms:=80 \
  decoder:=auto
```

Key parameters:
- `camera_password`: credential shared by both cameras (set per camera if needed
  by editing the launch file or using a YAML override).
- `transport`: `udp` gives the lowest latency on a clean LAN; fall back to `tcp`
  if UDP packets are blocked.
- `latency_ms`: jitter buffer requested from the camera; values between 40 and 120
  balance burst tolerance and latency.
- `decoder`: `auto` picks `nvh264dec`, `vaapih264dec`, `v4l2h264dec`, then
  `avdec_h264` as a fallback. Override to force a specific element.

The node publishes:
- `/camera/camera_1/image_raw`
- `/camera/camera_2/image_raw`

Each publisher is configured with `queue_size=1` and `tcp_nodelay=True`, so only
fresh frames are delivered to ROS subscribers.

## 4. Camera Configuration Profiles

Apply the following via the VIGI web UI (Configuration → Video & Audio):

```
Profile                Resolution  FPS  GOP  Codec  Bitrate Mode  Max Bitrate  Notes
------------------------------------------------------------------------------------
Ultra-Low-Latency      640x360     15   15   H.264  CBR          1.5 Mbps     Smart Coding OFF, B-frames OFF
Balanced Ops           896x512     15   30   H.264  CBR          3.0 Mbps     Smart Coding OFF, 3D DNR Low
High Quality (main)   2560x1440    15   30   H.264  VBR          6.0 Mbps     Used for recording; not for live control
```

Common toggles for all profiles:
- `H.264 Profile`: Baseline (if available) or Main.
- `Smart Coding / Smart Encoding`: Off.
- `Smart IR / WDR / HLC / BLC`: Off unless required for lighting; they add ISP delay.
- `SVC`: Off.
- `Audio`: Disabled.
- `3D DNR`: Low or Off (keeps sensor latency down).
- `I-Frame Interval`: equal to FPS (e.g., 15 fps → 15).
- `Max Bitrate`: keep conservative to avoid queueing in the camera encoder.
- After applying, reboot each camera to ensure RTSP uses the new GOP parameters.

## 5. Latency Verification Checklist

1. Place a high-frequency visual timer (LED or phone stopwatch) in the field of view.
2. Run the ROS node and subscribe via `rosrun image_view image_view _image_transport:=raw`.
3. Record both the physical timer and the ROS display using a smartphone at 60 fps.
4. In slow motion, count the frame difference between the live timer and the ROS view.
   Multiply frames by 16.7 ms to obtain the glass-to-glass latency. Target ≤ 24 frames
   (~400 ms) for the Ultra-Low-Latency profile.
5. Log `rostopic hz /camera/camera_1/image_raw` for at least 30 seconds; expect
   10–15 fps with ±0.5 fps jitter.
6. Monitor `GST_DEBUG=2` logs (optional) for `rtspsrc` jitter buffer underruns;
   if frequent, raise `latency_ms` to 100.

## 6. Troubleshooting

- **Stalled stream**: ensure IGMP snooping is enabled on the Netgear GS108Tv3 switch
  and that the Archer AX53 has SIP ALG disabled; both settings reduce RTSP teardown
  hiccups.
- **Residual >400 ms latency**: confirm the camera GOP and Smart Coding settings; a
  GOP of 60 or B-frames >0 will force the decoder to queue frames.
- **Dropped frames on UDP**: increase `latency_ms` in small steps (80 → 100 → 120) or
  switch to `transport:=tcp`.
- **CPU saturation**: use hardware decode by setting `decoder:=nvh264dec` (NVIDIA) or
  `decoder:=vaapih264dec` (Intel). Verify the plugin is installed with
  `gst-inspect-1.0 nvh264dec`.
- **Password management**: keep credentials out of version control; pass them via
  the launch argument or a private YAML file.

## 7. Legacy OpenCV Node

`scripts/ip_camera_streamer.py` is retained for reference but it relies on
OpenCV's FFmpeg bindings and incurs ~1 s extra latency due to unavoidable
internal buffering. Use `low_latency_rtsp_streamer.py` for production deployments.

## 8. License

MIT License
