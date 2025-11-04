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
        self.base_height = float(rospy.get_param("~spectator_height", 65.0))
        self.bev_yaw_deg = float(rospy.get_param("~spectator_yaw_deg", -90.0))
        self.bev_pitch_deg = float(rospy.get_param("~spectator_pitch_deg", -50.0))
        self.offset_x = float(rospy.get_param("~spectator_offset_x", -16.0))
        self.offset_y = float(rospy.get_param("~spectator_offset_y", 35.0))
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
        self.goal_duration_s = float(rospy.get_param("~goal_duration_s", 2.0))
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
            {"name": "goal2", "x":  58.000, "y": -41.500, "yaw": float(rospy.get_param("~goal2_yaw_deg", 0.0))},
            {"name": "goal3", "x": -45.810, "y":  -5.870, "yaw": float(rospy.get_param("~goal3_yaw_deg", -90.0))},
            {"name": "goal4", "x":  46.400, "y":  -5.890, "yaw": float(rospy.get_param("~goal4_yaw_deg", -90.0))},
            {"name": "goal5", "x":   0.000, "y": -30.000, "yaw": float(rospy.get_param("~goal5_yaw_deg", -90.0))},
        ]
        # Optional per-destination camera overrides (position + orientation)
        # Defaults for per-destination camera overrides (so launch params are optional)
        _override_defaults = {
            1: {"use_cam": True, "cam_x": -33.210, "cam_y": -43.790, "cam_z": 20.0, "cam_yaw_deg": 180.0, "cam_pitch_deg": -30.0},
            2: {"use_cam": True, "cam_x":  36.780, "cam_y": -43.670, "cam_z": 20.0, "cam_yaw_deg":   0.0, "cam_pitch_deg": -30.0},
            3: {"use_cam": True, "cam_x": -44.700, "cam_y":   8.240, "cam_z": 20.0, "cam_yaw_deg": -90.0, "cam_pitch_deg": -40.0},
            4: {"use_cam": True, "cam_x":  45.750, "cam_y":   7.520, "cam_z": 20.0, "cam_yaw_deg": -90.0, "cam_pitch_deg": -50.0},
            5: {"use_cam": True, "cam_x":   8.060, "cam_y": -15.820, "cam_z": 10.0, "cam_yaw_deg": -90.0, "cam_pitch_deg": -30.0},
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

        # Compute and apply default BEV
        self._compute_and_apply_default_view()

        # Keep default view locked when idle
        self._lock_timer = rospy.Timer(
            rospy.Duration(1.0 / max(0.1, self.lock_rate_hz)), self._lock_cb
        )

        # Subscriptions
        rospy.Subscriber("/selected_vehicle", String, self._selected_cb)
        rospy.Subscriber("/click_button", String, self._click_cb)
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

    def _goal_cb(self, msg: PoseStamped, role: str):
        # Cancel follow if running
        if self._follow_timer is not None:
            try:
                self._follow_timer.shutdown()
            except Exception:
                pass
            self._follow_timer = None

        self.mode = "goal"
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(self.goal_height)
        # Compute nearest preset and decide camera pose
        yaw_to_use = self.goal_yaw_deg
        best = None
        best_d = None
        try:
            for p in self._goal_presets:
                dxp = float(p["x"]) - x
                dyp = float(p["y"]) - y
                d = math.hypot(dxp, dyp)
                if best is None or d < best_d:
                    best = p
                    best_d = d
            if best is not None and best_d is not None and best_d <= self.goal_match_tolerance:
                yaw_to_use = float(best["yaw"])
        except Exception:
            pass

        use_override = bool(best is not None and best_d is not None and best_d <= self.goal_match_tolerance and best.get("use_cam", False))
        if use_override:
            tx = float(best.get("cam_x", x))
            ty = float(best.get("cam_y", y))
            tz = float(best.get("cam_z", z))
            yaw_used = float(best.get("cam_yaw_deg", yaw_to_use))
            pitch_used = float(best.get("cam_pitch_deg", self.goal_pitch_deg))
            preset_name = best.get("name", "unknown")
        else:
            tx = x
            ty = y
            tz = z
            yaw_used = yaw_to_use
            pitch_used = self.goal_pitch_deg
            preset_name = best.get("name", "default") if best is not None else "default"

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


def main():
    try:
        CamPerspective()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()



