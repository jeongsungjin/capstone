#!/usr/bin/env python3

import threading
from typing import Dict, List, Tuple, Optional

import rospy
from std_msgs.msg import Header
from carla_multi_vehicle_control.msg import TrafficLightPhase
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry

try:
    import pygame
except Exception as exc:
    pygame = None


class TrafficLightMonitorUI:
    def __init__(self) -> None:
        rospy.init_node("traffic_light_monitor_ui", anonymous=True)
        if pygame is None:
            raise RuntimeError("pygame is required for tl_monitor_ui")

        self.width = int(rospy.get_param("~width", 1200))
        self.height = int(rospy.get_param("~height", 800))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 30.0))
        self.panel_w = int(rospy.get_param("~panel_width", 360))
        self.panel_h = int(rospy.get_param("~panel_height", 220))
        self.panel_margin = int(rospy.get_param("~panel_margin", 16))
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.tl_stop_distance_m = float(rospy.get_param("~tl_stop_distance_m", 15.0))

        self._lock = threading.RLock()
        self._intersections: Dict[str, Dict] = {}
        # vehicle states: role -> {"pos":(x,y), "speed":float}
        self._vehicles: Dict[str, Dict] = {}

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Traffic Light Monitor")
        self.font = pygame.font.SysFont("arial", 16)
        self.clock = pygame.time.Clock()

        self.BLACK = (0, 0, 0)
        self.WHITE = (240, 240, 240)
        self.GREY = (90, 90, 90)
        self.DARK = (30, 30, 30)
        self.RED = (220, 60, 60)
        self.YELLOW = (240, 210, 60)
        self.GREEN = (60, 200, 90)
        self.CYAN = (60, 170, 210)

        rospy.Subscriber("/traffic_phase", TrafficLightPhase, self._phase_cb, queue_size=10)
        self._running = True

        # Subscribe vehicles (odometry + command)
        for idx in range(self.num_vehicles):
            role = f"ego_vehicle_{idx+1}"
            self._vehicles[role] = {"pos": None, "speed": None}
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)
            rospy.Subscriber(f"/carla/{role}/vehicle_control_cmd", AckermannDrive, self._cmd_cb, callback_args=role, queue_size=10)

    def _phase_cb(self, msg: TrafficLightPhase) -> None:
        with self._lock:
            ap_list = []
            for ap in msg.approaches:
                ap_list.append(
                    {
                        "name": ap.name,
                        "color": int(ap.color),
                        "rect": (float(ap.xmin), float(ap.xmax), float(ap.ymin), float(ap.ymax)),
                        "stop": (float(ap.stopline_x), float(ap.stopline_y)),
                    }
                )
            self._intersections[msg.intersection_id or "default"] = {
                "phase": msg.phase or "",
                "stamp": msg.header.stamp,
                "approaches": ap_list,
            }

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        with self._lock:
            st = self._vehicles.get(role)
            if st is not None:
                st["pos"] = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def _cmd_cb(self, msg: AckermannDrive, role: str) -> None:
        with self._lock:
            st = self._vehicles.get(role)
            if st is not None:
                st["speed"] = float(msg.speed)

    def _draw_panel(self, x: int, y: int, w: int, h: int, title: str, phase: str, stamp: rospy.Time, approaches: List[Dict]) -> None:
        pygame.draw.rect(self.screen, self.GREY, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, self.DARK, (x + 2, y + 2, w - 4, h - 4), border_radius=8)

        title_surf = self.font.render(f"{title}", True, self.WHITE)
        phase_surf = self.font.render(f"phase: {phase}", True, self.CYAN)
        ts = stamp.to_sec() if isinstance(stamp, rospy.Time) else 0.0
        time_surf = self.font.render(f"t={ts:.1f}", True, self.WHITE)
        self.screen.blit(title_surf, (x + 12, y + 10))
        self.screen.blit(phase_surf, (x + 12, y + 32))
        self.screen.blit(time_surf, (x + w - 100, y + 10))

        ax = x + 12
        ay = y + 60
        aw = w - 24
        ah = h - 72
        # split upper half for approaches, lower half for vehicles
        ah_top = int(ah * 0.55)
        ah_bot = ah - ah_top - 6
        pygame.draw.rect(self.screen, self.GREY, (ax, ay, aw, ah_top), width=1)

        row_h = 24
        for i, ap in enumerate(approaches[: max(1, int(ah_top / row_h))]):
            ry = ay + i * row_h
            pygame.draw.rect(self.screen, self.DARK, (ax + 2, ry + 2, aw - 4, row_h - 4))
            c = int(ap.get("color", 0))
            col = self.RED if c == 0 else (self.YELLOW if c == 1 else self.GREEN)
            pygame.draw.circle(self.screen, col, (ax + 16, ry + row_h // 2), 8)
            name = str(ap.get("name", ""))
            rect = ap.get("rect", (0, 0, 0, 0))
            stop = ap.get("stop", (0, 0))
            label = self.font.render(f"{name}", True, self.WHITE)
            meta = self.font.render(f"rect=({rect[0]:.1f},{rect[1]:.1f},{rect[2]:.1f},{rect[3]:.1f}) stop=({stop[0]:.1f},{stop[1]:.1f})", True, self.WHITE)
            self.screen.blit(label, (ax + 36, ry + 4))
            self.screen.blit(meta, (ax + 36, ry + 4 + 14))

        # Vehicles section
        vy = ay + ah_top + 6
        pygame.draw.rect(self.screen, self.GREY, (ax, vy, aw, ah_bot), width=1)
        vtitle = self.font.render("vehicles in area", True, self.CYAN)
        self.screen.blit(vtitle, (ax + 6, vy + 4))
        v_row_h = 22
        v_list = self._collect_vehicles_for_intersection(approaches)
        for j, row in enumerate(v_list[: max(1, int((ah_bot - 24) / v_row_h))]):
            ryy = vy + 24 + j * v_row_h
            pygame.draw.rect(self.screen, self.DARK, (ax + 2, ryy + 2, aw - 4, v_row_h - 4))
            # color lamp: current signal color at vehicle
            col = self.RED if row["color"] == 0 else (self.YELLOW if row["color"] == 1 else self.GREEN)
            pygame.draw.circle(self.screen, col, (ax + 16, ryy + v_row_h // 2), 6)
            txt = self.font.render(
                f"{row['role']}  d_stop={row['dist']:.1f}m  cmd_v={row['speed']:.2f}  {row['decision']}",
                True,
                self.WHITE,
            )
            self.screen.blit(txt, (ax + 32, ryy + 3))

    def _collect_vehicles_for_intersection(self, approaches: List[Dict]) -> List[Dict]:
        """Return list of dicts: role,color,dist,speed,decision for vehicles inside these approaches."""
        out: List[Dict] = []
        with self._lock:
            vehicles = dict(self._vehicles)
        # Build flat approach list with color and stopline
        aps: List[Tuple[Tuple[float, float, float, float], Tuple[float, float], int]] = []
        for ap in approaches:
            rect = ap.get("rect", (0.0, 0.0, 0.0, 0.0))
            stop = ap.get("stop", (0.0, 0.0))
            color = int(ap.get("color", 0))
            aps.append((rect, stop, color))
        for role, st in vehicles.items():
            pos = st.get("pos")
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            speed = float(st.get("speed") or 0.0)
            # find containing approach
            best = None
            for rect, stop, color in aps:
                xmin, xmax, ymin, ymax = rect
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                    dx = stop[0] - x
                    dy = stop[1] - y
                    dist = (dx * dx + dy * dy) ** 0.5
                    # decision
                    if color == 0 and dist <= self.tl_stop_distance_m:
                        decision = "STOP"
                    elif color == 1 and dist <= self.tl_stop_distance_m * 0.7:
                        decision = "CAUTIOUS-STOP"
                    else:
                        decision = "GO"
                    best = {"role": role, "color": color, "dist": dist, "speed": speed, "decision": decision}
                    break
            if best:
                out.append(best)
        # sort by dist ascending
        out.sort(key=lambda r: r["dist"])
        return out

    def run(self) -> None:
        while not rospy.is_shutdown() and self._running:
            self.clock.tick(max(5.0, self.refresh_hz))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
            self.screen.fill(self.BLACK)
            with self._lock:
                items = list(self._intersections.items())
            cols = max(1, self.width // (self.panel_w + self.panel_margin))
            for idx, (iid, data) in enumerate(items):
                col = idx % cols
                row = idx // cols
                px = self.panel_margin + col * (self.panel_w + self.panel_margin)
                py = self.panel_margin + row * (self.panel_h + self.panel_margin)
                self._draw_panel(
                    px,
                    py,
                    self.panel_w,
                    self.panel_h,
                    f"{iid}",
                    str(data.get("phase", "")),
                    data.get("stamp", rospy.Time(0)),
                    data.get("approaches", []),
                )
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    try:
        ui = TrafficLightMonitorUI()
        ui.run()
    except Exception as e:
        rospy.logfatal("tl_monitor_ui crashed: %s", e)
        raise


