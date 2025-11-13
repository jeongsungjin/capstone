#!/usr/bin/env python3

import math
import sys
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

try:
    # Ensure CARLA API on path via helper
    from setup_carla_path import CARLA_BUILD_PATH  # noqa: F401 (side effect: sys.path insert)
except Exception:
    pass

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"CARLA import failed in cam_perspective: {exc}")
    carla = None


class CamPerspective:
    def __init__(self):
        rospy.init_node("cam_perspective", anonymous=False)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        # Parameters
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 4))

        # Default BEV view parameters (matching prior behavior)
        self.auto_height = bool(rospy.get_param("~spectator_auto_height", False))
        self.min_height = float(rospy.get_param("~spectator_min_height", 65.0))
        self.base_height = float(rospy.get_param("~spectator_height", 185.0))
        self.bev_yaw_deg = float(rospy.get_param("~spectator_yaw_deg", -90.0))
        self.bev_pitch_deg = float(rospy.get_param("~spectator_pitch_deg", -60.0))
        self.offset_x = float(rospy.get_param("~spectator_offset_x", -12.0))
        self.offset_y = float(rospy.get_param("~spectator_offset_y", 95.0))
        self.view_right_m = float(rospy.get_param("~spectator_view_right_m", 0.0))
        self.view_up_m = float(rospy.get_param("~spectator_view_up_m", 0.0))
        self.lock_rate_hz = float(rospy.get_param("~lock_rate_hz", 5.0))
        # Ignore initial latched /selected_vehicle for a short window to keep BEV at startup
        self.selected_ignore_window_s = float(rospy.get_param("~selected_ignore_window_s", 1.0))

        # Follow-view parameters
        self.follow_duration_s = float(rospy.get_param("~follow_duration_s", 3.0))
        self.follow_rate_hz = float(rospy.get_param("~follow_rate_hz", 20.0))
        self.follow_back_m = float(rospy.get_param("~follow_back_m", 10.0))
        self.follow_up_m = float(rospy.get_param("~follow_up_m", 8.0))
        self.follow_right_m = float(rospy.get_param("~follow_right_m", -4.0))
        # Smoothing factors (0.0=즉시, 1.0=전혀 이동 안 함)
        self.follow_pos_alpha = float(rospy.get_param("~follow_pos_alpha", 0.2))
        self.follow_ang_alpha = float(rospy.get_param("~follow_ang_alpha", 0.25))

        # Goal-view parameters
        self.goal_duration_s = float(rospy.get_param("~goal_duration_s", 4.0))
        self.goal_height = float(rospy.get_param("~goal_height", 5.0))
        self.goal_yaw_deg = float(rospy.get_param("~goal_yaw_deg", -90.0))
        self.goal_pitch_deg = float(rospy.get_param("~goal_pitch_deg", 0.0))
        self.goal_rate_hz = float(rospy.get_param("~goal_rate_hz", 30.0))
        self.goal_pos_alpha = float(rospy.get_param("~goal_pos_alpha", 0.25))
        self.goal_ang_alpha = float(rospy.get_param("~goal_ang_alpha", 0.3))
        # Preset matching: map specific destination points to their own yaw
        self.goal_match_tolerance = float(rospy.get_param("~goal_match_tolerance", 3.0))
        self._goal_presets = [
            {"name": "goal1", "x": -58.010, "y": -42.790, "yaw": float(rospy.get_param("~goal1_yaw_deg", 180.0))},
            {"name": "goal2", "x": -41.980, "y": -54.270, "yaw": float(rospy.get_param("~goal2_yaw_deg", 0.0))},
            {"name": "goal3", "x":   0.000, "y": -30.000, "yaw": float(rospy.get_param("~goal3_yaw_deg", -90.0))},
            {"name": "goal4", "x":  43.090, "y": -53.940, "yaw": float(rospy.get_param("~goal4_yaw_deg", -90.0))},
            {"name": "goal5", "x":  58.250, "y": -37.830, "yaw": float(rospy.get_param("~goal5_yaw_deg", -90.0))},
            {"name": "goal6", "x":  42.520, "y":  -6.030,  "yaw": float(rospy.get_param("~goal6_yaw_deg", -90.0))},
            {"name": "goal7", "x": -35.870, "y":  -2.040,  "yaw": float(rospy.get_param("~goal7_yaw_deg", -90.0))},
            {"name": "goal8", "x":  20.000, "y":  -1.750,  "yaw": float(rospy.get_param("~goal8_yaw_deg", -90.0))},
        ]
        # Optional per-destination camera overrides (position + orientation)
        # Defaults for per-destination camera overrides (so launch params are optional)
        _override_defaults = {
            1: {"use_cam": True, "cam_x":   1.650, "cam_y":  2.810, "cam_z": 80.0, "cam_yaw_deg": 220.0, "cam_pitch_deg": -35.0},
            2: {"use_cam": True, "cam_x":  -41.980, "cam_y": 1.460, "cam_z": 70.0, "cam_yaw_deg":  -90.0, "cam_pitch_deg": -40.0},
            3: {"use_cam": True, "cam_x":   0.000, "cam_y":  6.180, "cam_z": 60.0, "cam_yaw_deg": -90.0, "cam_pitch_deg": -50.0},
            4: {"use_cam": True, "cam_x":  43.090, "cam_y": -21.460, "cam_z": 60.0, "cam_yaw_deg": -90.0, "cam_pitch_deg": -50.0},
            5: {"use_cam": True, "cam_x":  7.310, "cam_y": -35.620, "cam_z": 70.0, "cam_yaw_deg":  0.0, "cam_pitch_deg": -40.0},
            6: {"use_cam": True, "cam_x":  42.520, "cam_y":  10.800, "cam_z": 30.0, "cam_yaw_deg": -90.0, "cam_pitch_deg": -43.0},
            7: {"use_cam": True, "cam_x": -31.150, "cam_y": -58.830, "cam_z": 60.0, "cam_yaw_deg": 90.0, "cam_pitch_deg": -35.0},
            8: {"use_cam": True, "cam_x":  17.200, "cam_y": -38.110, "cam_z": 60.0, "cam_yaw_deg": 90.0, "cam_pitch_deg": -45.0},
        }
        for idx, p in enumerate(self._goal_presets, start=1):
            d = _override_defaults.get(idx, None)
            p["use_cam"] = bool(rospy.get_param(f"~goal{idx}_use_cam", d["use_cam"] if d else False))
            p["cam_x"] = float(rospy.get_param(f"~goal{idx}_cam_x", d["cam_x"] if d else p["x"]))
            p["cam_y"] = float(rospy.get_param(f"~goal{idx}_cam_y", d["cam_y"] if d else p["y"]))
            p["cam_z"] = float(rospy.get_param(f"~goal{idx}_cam_z", d["cam_z"] if d else self.goal_height))
            p["cam_yaw_deg"] = float(rospy.get_param(f"~goal{idx}_cam_yaw_deg", d["cam_yaw_deg"] if d else p["yaw"]))
            p["cam_pitch_deg"] = float(rospy.get_param(f"~goal{idx}_cam_pitch_deg", d["cam_pitch_deg"] if d else self.goal_pitch_deg))

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.spectator = self.world.get_spectator()
        
        
                # === Sun direction control (Directional Light equivalent) ===
        # Allow manual adjustment via ROS params or code
        self.sun_pitch_deg = float(rospy.get_param("~sun_pitch_deg", -130.0))    # Unreal Rotation X
        self.sun_yaw_deg = float(rospy.get_param("~sun_yaw_deg", 120.0))     # Unreal Rotation Y
        self.sun_roll_deg = float(rospy.get_param("~sun_roll_deg", 120.0))   # Unreal Rotation Z (not used, but logged)

        weather = self.world.get_weather()
        # Unreal uses Pitch (down is negative) → CARLA uses positive Altitude (up)
        weather.sun_altitude_angle = -self.sun_pitch_deg
        weather.sun_azimuth_angle = self.sun_yaw_deg
        self.world.set_weather(weather)

        rospy.loginfo(
            "☀️  Sun direction set from Unreal rotation: pitch=%.1f, yaw=%.1f, roll=%.1f → altitude=%.1f, azimuth=%.1f",
            self.sun_pitch_deg, self.sun_yaw_deg, self.sun_roll_deg,
            weather.sun_altitude_angle, weather.sun_azimuth_angle
        )

        # State
        self.mode = "default"  # default|follow|goal
        self._default_transform = None
        self._lock_timer = None
        self._follow_timer = None
        self._goal_timer = None
        self._follow_end_time = None  # follow persists until /click_button
        self._follow_role = None
        self._follow_cam = None  # {'x','y','z','yaw','pitch'}
        self._goal_cam = None    # {'x','y','z','yaw','pitch'} for goal animation
        self._goal_target = None # {'x','y','z','yaw','pitch'} target for goal animation
        self._goal_anim_timer = None
        self._selected_ignore_until = rospy.Time.now().to_sec() + max(0.0, self.selected_ignore_window_s)
        # Destination-driven camera override (1..8), set by /destination
        self._forced_use_cam_index = None
        self._destination_to_cam_index = {
            "hotel": 1,
            "building": 2,
            "office": 3,
            "school": 4,
            "home": 5,
            "church": 6,
            "mart": 7,
            "hospital": 8,
        }

        # Compute and apply default BEV
        self._compute_and_apply_default_view()

        # Keep default view locked when idle
        self._lock_timer = rospy.Timer(
            rospy.Duration(1.0 / max(0.1, self.lock_rate_hz)), self._lock_cb
        )

        # Subscriptions
        rospy.Subscriber("/selected_vehicle", String, self._selected_cb)
        rospy.Subscriber("/click_button", String, self._click_cb)
        rospy.Subscriber("/destination", String, self._destination_cb)
        for i in range(1, self.num_vehicles + 1):
            role = f"ego_vehicle_{i}"
            topic = f"/override_goal/{role}"
            rospy.Subscriber(topic, PoseStamped, self._goal_cb, callback_args=role)

        rospy.on_shutdown(self._on_shutdown)

    def _compute_and_apply_default_view(self):
        spawn_points = self.carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available to compute BEV view")

        xs = [sp.location.x for sp in spawn_points]
        ys = [sp.location.y for sp in spawn_points]
        cx = sum(xs) / float(len(xs))
        cy = sum(ys) / float(len(ys))
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        height = self.base_height
        if self.auto_height:
            height = max(self.min_height, max(span_x, span_y) * 1.1)

        target_x = cx + self.offset_x
        target_y = cy + self.offset_y
        if self.view_right_m != 0.0 or self.view_up_m != 0.0:
            yaw_rad = math.radians(self.bev_yaw_deg)
            rx = math.cos(yaw_rad)
            ry = math.sin(yaw_rad)
            ux = -math.sin(yaw_rad)
            uy = math.cos(yaw_rad)
            target_x += self.view_right_m * rx + self.view_up_m * ux
            target_y += self.view_right_m * ry + self.view_up_m * uy

        self._default_transform = carla.Transform(
            carla.Location(x=target_x, y=target_y, z=height),
            carla.Rotation(pitch=self.bev_pitch_deg, yaw=self.bev_yaw_deg, roll=0.0),
        )
        self._apply_transform(self._default_transform)
        rospy.loginfo(
            "cam_perspective: default BEV at (%.1f, %.1f, %.1f) yaw=%.1f span=(%.1f, %.1f)",
            target_x,
            target_y,
            height,
            self.bev_yaw_deg,
            span_x,
            span_y,
        )

    def _apply_transform(self, transform):
        try:
            self.spectator.set_transform(transform)
        except Exception:
            pass

    def _lock_cb(self, _event):
        if self.mode != "default" or self._default_transform is None:
            return
        self._apply_transform(self._default_transform)

    def _selected_cb(self, msg: String):
        role = (msg.data or "").strip()
        if not role:
            return
        # Skip early latched message to keep BEV at startup
        if rospy.Time.now().to_sec() < self._selected_ignore_until:
            rospy.loginfo("cam_perspective: ignoring early /selected_vehicle '%s' to keep BEV", role)
            return
        rospy.loginfo("cam_perspective: selected %s", role)
        self._start_follow(role)

    def _start_follow(self, role: str):
        self.mode = "follow"
        # No timeout – will end on /click_button
        self._follow_end_time = None
        self._follow_role = role
        # Initialize smoothing state from current spectator transform
        try:
            t = self.spectator.get_transform()
            self._follow_cam = {
                'x': float(t.location.x),
                'y': float(t.location.y),
                'z': float(t.location.z),
                'yaw': float(t.rotation.yaw),
                'pitch': float(t.rotation.pitch),
            }
        except Exception:
            self._follow_cam = None
        if self._goal_anim_timer is not None:
            try:
                self._goal_anim_timer.shutdown()
            except Exception:
                pass
            self._goal_anim_timer = None
        self._goal_cam = None
        self._goal_target = None
        if self._goal_timer is not None:
            try:
                self._goal_timer.shutdown()
            except Exception:
                pass
            self._goal_timer = None

        # (Re)start follow timer with current role
        if self._follow_timer is not None:
            try:
                self._follow_timer.shutdown()
            except Exception:
                pass
        self._follow_timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.follow_rate_hz)),
            self._follow_cb,
            oneshot=False,
            reset=True,
        )

    def _follow_cb(self, _event):
        if self._follow_end_time is not None and time.time() >= self._follow_end_time:
            try:
                if self._follow_timer is not None:
                    self._follow_timer.shutdown()
            except Exception:
                pass
            self._follow_timer = None
            self.mode = "default"
            self._follow_cam = None
            return

        role = self._follow_role
        if not role:
            return

        # Find target vehicle by role
        actors = self.world.get_actors().filter("vehicle.*")
        target = None
        for actor in actors:
            if actor.attributes.get("role_name", "") == role:
                target = actor
                break
        if target is None:
            return

        t = target.get_transform()
        yaw_rad = math.radians(t.rotation.yaw)

        # Local-space offsets: back, right, up
        bx = -math.cos(yaw_rad) * self.follow_back_m
        by = -math.sin(yaw_rad) * self.follow_back_m
        rx = -math.sin(yaw_rad) * self.follow_right_m
        ry = math.cos(yaw_rad) * self.follow_right_m
        cam_x = t.location.x + bx + rx
        cam_y = t.location.y + by + ry
        cam_z = t.location.z + self.follow_up_m

        # Look-at vehicle
        dx = t.location.x - cam_x
        dy = t.location.y - cam_y
        dz = t.location.z - cam_z
        yaw = math.degrees(math.atan2(dy, dx))
        dist_xy = max(1e-3, math.hypot(dx, dy))
        # In CARLA(Unreal) down-look is negative pitch; ensure camera looks down when above target
        pitch = math.degrees(math.atan2(dz, dist_xy))
        # Initialize smoothing state if needed
        if self._follow_cam is None:
            self._follow_cam = {'x': cam_x, 'y': cam_y, 'z': cam_z, 'yaw': yaw, 'pitch': pitch}

        # Smooth towards target
        def _lerp(a, b, alpha):
            return a + max(0.0, min(1.0, alpha)) * (b - a)

        def _lerp_deg(a, b, alpha):
            # shortest-path interpolation in degrees
            delta = (b - a + 180.0) % 360.0 - 180.0
            return a + max(0.0, min(1.0, alpha)) * delta

        nx = _lerp(self._follow_cam['x'], cam_x, self.follow_pos_alpha)
        ny = _lerp(self._follow_cam['y'], cam_y, self.follow_pos_alpha)
        nz = _lerp(self._follow_cam['z'], cam_z, self.follow_pos_alpha)
        nyaw = _lerp_deg(self._follow_cam['yaw'], yaw, self.follow_ang_alpha)
        npitch = _lerp_deg(self._follow_cam['pitch'], pitch, self.follow_ang_alpha)

        self._follow_cam.update({'x': nx, 'y': ny, 'z': nz, 'yaw': nyaw, 'pitch': npitch})

        self._apply_transform(
            carla.Transform(
                carla.Location(x=nx, y=ny, z=nz),
                carla.Rotation(pitch=npitch, yaw=nyaw, roll=0.0),
            )
        )
        rospy.loginfo_throttle(0.5, "cam_perspective: following %s at (%.1f, %.1f, %.1f) yaw=%.1f pitch=%.1f → tgt (%.1f, %.1f, %.1f) yaw=%.1f pitch=%.1f",
                               role, nx, ny, nz, nyaw, npitch, cam_x, cam_y, cam_z, yaw, pitch)

    def _click_cb(self, _msg: String):
        # End follow/goal and revert to default BEV immediately
        self.mode = "default"
        # stop follow timer
        if self._follow_timer is not None:
            try:
                self._follow_timer.shutdown()
            except Exception:
                pass
            self._follow_timer = None
        self._follow_cam = None
        # stop goal timers/animation
        if self._goal_timer is not None:
            try:
                self._goal_timer.shutdown()
            except Exception:
                pass
            self._goal_timer = None
        if self._goal_anim_timer is not None:
            try:
                self._goal_anim_timer.shutdown()
            except Exception:
                pass
            self._goal_anim_timer = None
        self._goal_cam = None
        self._goal_target = None
        # apply default view
        if self._default_transform is not None:
            self._apply_transform(self._default_transform)

    def _start_smooth_transition(self, tx: float, ty: float, tz: float, yaw_deg: float, pitch_deg: float, mode_str: str):
        # Set mode (e.g., "goal" for destination, "default" for BEV)
        self.mode = mode_str
        # Stop any existing animation
        if self._goal_anim_timer is not None:
            try:
                self._goal_anim_timer.shutdown()
            except Exception:
                pass
            self._goal_anim_timer = None
        # Initialize current camera state from spectator
        try:
            tcur = self.spectator.get_transform()
            self._goal_cam = {
                'x': float(tcur.location.x),
                'y': float(tcur.location.y),
                'z': float(tcur.location.z),
                'yaw': float(tcur.rotation.yaw),
                'pitch': float(tcur.rotation.pitch),
            }
        except Exception:
            self._goal_cam = {'x': tx, 'y': ty, 'z': tz, 'yaw': yaw_deg, 'pitch': pitch_deg}
        # Set animation target
        self._goal_target = {'x': tx, 'y': ty, 'z': tz, 'yaw': yaw_deg, 'pitch': pitch_deg}
        # Start animation timer
        self._goal_anim_timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.goal_rate_hz)),
            self._goal_anim_cb,
            oneshot=False,
            reset=True,
        )

    def _destination_cb(self, msg: String):
        name = (msg.data or "").strip().lower()
        idx = self._destination_to_cam_index.get(name, None)
        if idx is not None:
            self._forced_use_cam_index = idx
            # Stop any ongoing follow/goal timers
            if self._follow_timer is not None:
                try:
                    self._follow_timer.shutdown()
                except Exception:
                    pass
                self._follow_timer = None
            if self._goal_timer is not None:
                try:
                    self._goal_timer.shutdown()
                except Exception:
                    pass
                self._goal_timer = None
            if self._goal_anim_timer is not None:
                try:
                    self._goal_anim_timer.shutdown()
                except Exception:
                    pass
                self._goal_anim_timer = None
            self._goal_cam = None
            self._goal_target = None
            # Prevent default lock from overriding this view
            # Smooth transition to the mapped preset viewpoint
            try:
                if 1 <= idx <= len(self._goal_presets):
                    preset = self._goal_presets[idx - 1]
                    tx = float(preset.get("cam_x", preset.get("x", 0.0)))
                    ty = float(preset.get("cam_y", preset.get("y", 0.0)))
                    tz = float(preset.get("cam_z", self.goal_height))
                    yaw_used = float(preset.get("cam_yaw_deg", preset.get("yaw", self.goal_yaw_deg)))
                    pitch_used = float(preset.get("cam_pitch_deg", self.goal_pitch_deg))
                    self._start_smooth_transition(tx, ty, tz, yaw_used, pitch_used, mode_str="goal")
                    rospy.loginfo("cam_perspective: destination '%s' → use_cam %d (preset=%s)",
                                  name, idx, preset.get("name", "unknown"))
                else:
                    rospy.logwarn("cam_perspective: destination index out of range: %d", idx)
            except Exception as exc:
                rospy.logwarn("cam_perspective: failed to apply destination view '%s' (idx=%d): %s", name, idx, str(exc))
        else:
            rospy.logwarn("cam_perspective: unknown destination '%s'", name)

    def _goal_cb(self, msg: PoseStamped, role: str):
        # Cancel follow if running
        if self._follow_timer is not None:
            try:
                self._follow_timer.shutdown()
            except Exception:
                pass
            self._follow_timer = None

        # Smoothly switch to default BEV view on goal
        self.mode = "default"
        # stop any goal timers/animation if previously running
        if self._goal_timer is not None:
            try:
                self._goal_timer.shutdown()
            except Exception:
                pass
            self._goal_timer = None
        # Start smooth animation to default transform
        if self._default_transform is not None:
            tx = float(self._default_transform.location.x)
            ty = float(self._default_transform.location.y)
            tz = float(self._default_transform.location.z)
            yaw_used = float(self._default_transform.rotation.yaw)
            pitch_used = float(self._default_transform.rotation.pitch)
            self._start_smooth_transition(tx, ty, tz, yaw_used, pitch_used, mode_str="default")
            rospy.loginfo("cam_perspective: smooth BEV transition on goal for %s", role)
        else:
            rospy.loginfo("cam_perspective: default transform unavailable; skipping smooth BEV")

        # Initialize smoothing state for goal animation from current spectator
        try:
            tcur = self.spectator.get_transform()
            self._goal_cam = {
                'x': float(tcur.location.x),
                'y': float(tcur.location.y),
                'z': float(tcur.location.z),
                'yaw': float(tcur.rotation.yaw),
                'pitch': float(tcur.rotation.pitch),
            }
        except Exception:
            self._goal_cam = {'x': tx, 'y': ty, 'z': tz, 'yaw': yaw_used, 'pitch': pitch_used}
        self._goal_target = {'x': tx, 'y': ty, 'z': tz, 'yaw': yaw_used, 'pitch': pitch_used}

        # Start goal animation timer towards target
        if self._goal_anim_timer is not None:
            try:
                self._goal_anim_timer.shutdown()
            except Exception:
                pass
        self._goal_anim_timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.goal_rate_hz)),
            self._goal_anim_cb,
            oneshot=False,
            reset=True,
        )

        rospy.loginfo(
            "cam_perspective: goal view at (%.1f, %.1f, %.1f) yaw=%.1f pitch=%.1f (preset=%s, d=%.2f, tol=%.2f, override=%s)",
            tx, ty, tz, yaw_used, pitch_used,
            preset_name,
            (best_d if best_d is not None else -1.0), self.goal_match_tolerance,
            str(use_override),
        )

        # Revert to default after duration
        def _revert(_evt):
            self.mode = "default"
            if self._goal_anim_timer is not None:
                try:
                    self._goal_anim_timer.shutdown()
                except Exception:
                    pass
                self._goal_anim_timer = None
            self._goal_cam = None
            self._goal_target = None

        if self._goal_timer is not None:
            try:
                self._goal_timer.shutdown()
            except Exception:
                pass
        self._goal_timer = rospy.Timer(
            rospy.Duration(max(0.0, self.goal_duration_s)), _revert, oneshot=True
        )

    def _on_shutdown(self):
        try:
            if self._lock_timer is not None:
                self._lock_timer.shutdown()
        except Exception:
            pass
        try:
            if self._follow_timer is not None:
                self._follow_timer.shutdown()
        except Exception:
            pass
        try:
            if self._goal_timer is not None:
                self._goal_timer.shutdown()
        except Exception:
            pass
        try:
            if self._goal_anim_timer is not None:
                self._goal_anim_timer.shutdown()
        except Exception:
            pass

    def _goal_anim_cb(self, _event):
        if self.mode != "goal" or self._goal_target is None or self._goal_cam is None:
            return
        tx = self._goal_target['x']
        ty = self._goal_target['y']
        tz = self._goal_target['z']
        tyaw = self._goal_target['yaw']
        tpitch = self._goal_target['pitch']

        def _lerp(a, b, alpha):
            return a + max(0.0, min(1.0, alpha)) * (b - a)

        def _lerp_deg(a, b, alpha):
            delta = (b - a + 180.0) % 360.0 - 180.0
            return a + max(0.0, min(1.0, alpha)) * delta

        nx = _lerp(self._goal_cam['x'], tx, self.goal_pos_alpha)
        ny = _lerp(self._goal_cam['y'], ty, self.goal_pos_alpha)
        nz = _lerp(self._goal_cam['z'], tz, self.goal_pos_alpha)
        nyaw = _lerp_deg(self._goal_cam['yaw'], tyaw, self.goal_ang_alpha)
        npitch = _lerp_deg(self._goal_cam['pitch'], tpitch, self.goal_ang_alpha)

        self._goal_cam.update({'x': nx, 'y': ny, 'z': nz, 'yaw': nyaw, 'pitch': npitch})
        self._apply_transform(
            carla.Transform(
                carla.Location(x=nx, y=ny, z=nz),
                carla.Rotation(pitch=npitch, yaw=nyaw, roll=0.0),
            )
        )
        # Stop animation when sufficiently close to target
        pos_err = math.hypot(math.hypot(nx - tx, ny - ty), nz - tz)
        ang_err = max(abs(((nyaw - tyaw + 180.0) % 360.0) - 180.0),
                      abs(((npitch - tpitch + 180.0) % 360.0) - 180.0))
        if pos_err <= 0.15 and ang_err <= 1.0:
            try:
                if self._goal_anim_timer is not None:
                    self._goal_anim_timer.shutdown()
            except Exception:
                pass
            self._goal_anim_timer = None
            self._goal_cam = None
            self._goal_target = None


def main():
    try:
        CamPerspective()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()



