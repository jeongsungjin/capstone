#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any

import rospy
from ackermann_msgs.msg import AckermannDrive
from capstone_msgs.msg import BEVInfo


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class PIDState:
    integ: float = 0.0
    prev_err: float = 0.0
    prev_t: float = 0.0


class BEVSpeedOverridePIDNode:
    """
    BEVInfo(speeds_mps) + raw AckermannDrive -> final AckermannDrive

    - Subscribes:
        * /bev_info_raw (capstone_msgs/BEVInfo)  : provides per-id measured speed (m/s)
        * /carla/ego_vehicle_{i}/vehicle_control_cmd_raw (ackermann_msgs/AckermannDrive)
    - Publishes:
        * /carla/ego_vehicle_{i}/vehicle_control_cmd (ackermann_msgs/AckermannDrive)

    Behavior:
    - Keeps steering_angle from RAW command.
    - Overrides speed with PID output (target_speed_mps - measured_speed_mps).
    - If BEV speed for a vehicle is missing, falls back to RAW speed (pass-through).
    - Per-vehicle params supported via:
        default/kp, default/ki, default/kd, default/target_speed_mps, default/min_speed_cmd, default/max_speed_cmd
        vehicles/ego_vehicle_1/kp ... (or loaded as dict via <rosparam param="vehicles"> ...)
    """

    def __init__(self) -> None:
        rospy.init_node("bev_speed_override_pid", anonymous=True)

        # --- Params ---
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.bev_topic = str(rospy.get_param("~bev_topic", "/bev_info_raw"))
        self.cmd_raw_template = str(rospy.get_param("~cmd_raw_template",
                                                   "/carla/ego_vehicle_{vid}/vehicle_control_cmd_raw"))
        self.cmd_out_template = str(rospy.get_param("~cmd_out_template",
                                                   "/carla/ego_vehicle_{vid}/vehicle_control_cmd"))
        # If incoming speeds_mps are scaled (e.g., 10x), apply a multiplier here.
        self.meas_speed_scale = float(rospy.get_param("~bev_speed_scale", 1.0))
        # Debug/logging controls
        self.log_missing_meas = bool(rospy.get_param("~log_missing_measurement", False))
        self.log_meas = bool(rospy.get_param("~log_measurement", False))

        # defaults (fallback)
        self.default_kp = float(rospy.get_param("~default/kp", 1.0))
        self.default_ki = float(rospy.get_param("~default/ki", 0.2))
        self.default_kd = float(rospy.get_param("~default/kd", 0.0))
        self.default_target = float(rospy.get_param("~default/target_speed_mps", 0.5))
        self.default_min_cmd = float(rospy.get_param("~default/min_speed_cmd", 0.0))
        self.default_max_cmd = float(rospy.get_param("~default/max_speed_cmd", 1.0))

        # mapping: how to interpret BEV ids
        self.use_detection_index = bool(rospy.get_param("~use_detection_index", True))
        self.id_offset = int(rospy.get_param("~id_offset", 0))

        # discontinuity/teleport safety
        self.enable_jump_reset = bool(rospy.get_param("~enable_jump_reset", True))
        self.pos_jump_m = float(rospy.get_param("~pos_jump_m", 1.0))
        self.max_integrator = float(rospy.get_param("~max_integrator", 2.0))
        self.freeze_integrator_if_saturated = bool(rospy.get_param("~freeze_integrator_if_saturated", True))

        # per-vehicle overrides dictionary (loaded from rosparam param="vehicles")
        self.vehicles_cfg: Dict[str, Dict[str, Any]] = rospy.get_param("~vehicles", {})

        # --- State ---
        self.meas_speed_mps: Dict[int, float] = {}          # vid -> speed (m/s)
        self.meas_xy: Dict[int, tuple] = {}                 # vid -> (x, y) for jump reset
        self.raw_cmd: Dict[int, AckermannDrive] = {}        # vid -> last raw command
        self.pid: Dict[int, PIDState] = {}                  # vid -> PID integrator state
        self.last_bev_stamp: Optional[rospy.Time] = None

        # --- Pub/Sub ---
        self.sub_bev = rospy.Subscriber(self.bev_topic, BEVInfo, self._bev_cb, queue_size=1)

        self.pubs: Dict[int, rospy.Publisher] = {}
        self.subs: Dict[int, rospy.Subscriber] = {}

        for vid in range(1, self.num_vehicles + 1):
            raw_topic = self.cmd_raw_template.format(vid=vid)
            out_topic = self.cmd_out_template.format(vid=vid)

            self.subs[vid] = rospy.Subscriber(raw_topic, AckermannDrive, self._raw_cb,
                                              callback_args=vid, queue_size=5)
            self.pubs[vid] = rospy.Publisher(out_topic, AckermannDrive, queue_size=5)

            rospy.loginfo("[bev_speed_override_pid] vid=%d raw=%s -> out=%s", vid, raw_topic, out_topic)

        # Publish loop (fixed rate) so we can keep output even if raw is noisy;
        # but we require raw present at least once per vehicle to output.
        self.pub_hz = float(rospy.get_param("~publish_hz", 30.0))
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.pub_hz)), self._tick)

        rospy.loginfo(
            "[bev_speed_override_pid] bev=%s num=%d defaults: kp=%.3f ki=%.3f kd=%.3f target=%.2f min=%.2f max=%.2f",
            self.bev_topic, self.num_vehicles, self.default_kp, self.default_ki, self.default_kd,
            self.default_target, self.default_min_cmd, self.default_max_cmd
        )

    # -------------------------
    # Param lookup helpers
    # -------------------------
    def _role_name(self, vid: int) -> str:
        return f"ego_vehicle_{vid}"

    def _get_cfg(self, vid: int, key: str, default: float) -> float:
        role = self._role_name(vid)
        cfg = self.vehicles_cfg.get(role, {})
        try:
            if key in cfg:
                return float(cfg[key])
        except Exception:
            pass
        return float(default)

    def _get_pid_gains(self, vid: int):
        kp = self._get_cfg(vid, "kp", self.default_kp)
        ki = self._get_cfg(vid, "ki", self.default_ki)
        kd = self._get_cfg(vid, "kd", self.default_kd)
        return kp, ki, kd

    def _get_target(self, vid: int) -> float:
        return self._get_cfg(vid, "target_speed_mps", self.default_target)

    def _get_limits(self, vid: int):
        mn = self._get_cfg(vid, "min_speed_cmd", self.default_min_cmd)
        mx = self._get_cfg(vid, "max_speed_cmd", self.default_max_cmd)
        if mx < mn:
            mx = mn
        return mn, mx

    def _reset_pid(self, vid: int) -> None:
        self.pid[vid] = PIDState(integ=0.0, prev_err=0.0, prev_t=0.0)

    # -------------------------
    # Callbacks
    # -------------------------
    def _bev_cb(self, msg: BEVInfo) -> None:
        # map detections -> speeds_mps
        n = min(len(msg.ids), len(getattr(msg, "speeds_mps", [])))
        for i in range(n):
            # Option A: follow detection order (teleporter uses index-based role assignment)
            if self.use_detection_index:
                vid = i + 1 + self.id_offset
            else:
                try:
                    vid = int(msg.ids[i]) + self.id_offset
                except Exception:
                    continue
            if vid < 1 or vid > self.num_vehicles:
                continue
            try:
                v = float(msg.speeds_mps[i]) * self.meas_speed_scale
            except Exception:
                continue
            self.meas_speed_mps[vid] = v

            # optional jump reset using positions
            if self.enable_jump_reset and i < len(msg.center_xs) and i < len(msg.center_ys):
                try:
                    x = float(msg.center_xs[i])
                    y = float(msg.center_ys[i])
                except Exception:
                    continue
                prev = self.meas_xy.get(vid)
                self.meas_xy[vid] = (x, y)
                if prev is not None:
                    dx = x - prev[0]
                    dy = y - prev[1]
                    if math.hypot(dx, dy) > self.pos_jump_m:
                        self._reset_pid(vid)

        self.last_bev_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def _raw_cb(self, msg: AckermannDrive, vid: int) -> None:
        # store last raw cmd
        self.raw_cmd[vid] = msg
        if vid not in self.pid:
            self.pid[vid] = PIDState(integ=0.0, prev_err=0.0, prev_t=rospy.get_time())

    # -------------------------
    # Core control loop
    # -------------------------
    def _compute_speed_cmd(self, vid: int, raw_speed: float) -> float:
        """
        If measured speed exists: PID to reach target.
        Else: pass-through raw speed.
        """
        if vid not in self.meas_speed_mps:
            if self.log_missing_meas:
                rospy.logwarn_throttle(1.0, "[bev_speed_pid] vid=%d no measurement; pass-through raw=%.3f", vid, raw_speed)
            return raw_speed  # fallback

        v_meas = float(self.meas_speed_mps.get(vid, raw_speed))
        target = self._get_target(vid)
        kp, ki, kd = self._get_pid_gains(vid)
        mn, mx = self._get_limits(vid)

        st = self.pid.get(vid)
        now = rospy.get_time()
        if st is None:
            st = PIDState(integ=0.0, prev_err=0.0, prev_t=now)
            self.pid[vid] = st

        dt = max(1e-3, now - (st.prev_t if st.prev_t > 0 else now))
        err = target - v_meas

        p = kp * err
        d = kd * ((err - st.prev_err) / dt)

        # First compute without updating integrator to check saturation
        u_unsat = p + ki * st.integ + d
        u_sat = clamp(u_unsat, mn, mx)

        # Anti-windup: freeze integrator if saturated (optional)
        if ki > 0.0:
            if not (self.freeze_integrator_if_saturated and (u_sat != u_unsat)):
                st.integ = clamp(st.integ + err * dt, -self.max_integrator, self.max_integrator)

        u = clamp(p + ki * st.integ + d, mn, mx)

        st.prev_err = err
        st.prev_t = now
        if self.log_meas:
            rospy.loginfo_throttle(
                1.0,
                "[bev_speed_pid] vid=%d target=%.3f meas=%.3f cmd=%.3f (p=%.3f, i=%.3f, d=%.3f)",
                vid,
                target,
                v_meas,
                u,
                p,
                ki * st.integ,
                d,
            )
        return float(u)

    def _tick(self, _evt) -> None:
        # For each vehicle: if we have a raw cmd, publish final cmd
        for vid in range(1, self.num_vehicles + 1):
            raw = self.raw_cmd.get(vid)
            if raw is None:
                continue

            # keep steering from raw, override speed
            raw_speed = float(getattr(raw, "speed", 0.0))
            speed_cmd = self._compute_speed_cmd(vid, raw_speed)

            out = AckermannDrive()
            out.steering_angle = float(getattr(raw, "steering_angle", 0.0))
            out.steering_angle_velocity = float(getattr(raw, "steering_angle_velocity", 0.0))
            out.speed = float(speed_cmd)
            out.acceleration = float(getattr(raw, "acceleration", 0.0))
            out.jerk = float(getattr(raw, "jerk", 0.0))

            self.pubs[vid].publish(out)

        rospy.logdebug_throttle(1.0, "[bev_speed_override_pid] publishing for %d vehicles", self.num_vehicles)


if __name__ == "__main__":
    try:
        BEVSpeedOverridePIDNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
