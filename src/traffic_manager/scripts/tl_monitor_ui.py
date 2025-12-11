#!/usr/bin/env python3

import math
import threading
from typing import Dict, List, Tuple

import rospy
from carla_multi_vehicle_control.msg import TrafficLightPhase
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry

try:
    import pygame
except Exception as exc:
    pygame = None


def get_light_states(phase_name: str) -> Dict[str, bool]:
    state = {
        "M_LR_R": True, "M_LR_Y": False, "M_LR_G": False, "M_LR_LEFT": False,
        "M_RL_R": True, "M_RL_Y": False, "M_RL_G": False,
        "S_R": True, "S_Y": False, "S_G": False,
    }
    if phase_name == "P1_MAIN_GREEN":
        state.update({
            "M_LR_R": True, "M_LR_G": False,
            "M_RL_R": False, "M_RL_G": True,
        })
    elif phase_name == "P1_YELLOW":
        state.update({
            "M_LR_R": True, "M_LR_Y": False,
            "M_RL_R": False, "M_RL_Y": True,
        })
    elif phase_name == "P2_MAIN_GREEN":
        state.update({
            "M_LR_R": False, "M_LR_LEFT": True,
            "M_RL_R": True, "M_LR_G": True,
        })
    elif phase_name == "P2_YELLOW":
        state.update({
            "M_LR_R": False, "M_LR_Y": True,
            "M_RL_R": True,
        })
    elif phase_name == "P3_SIDE_GREEN":
        state.update({"S_R": False, "S_G": True})
    elif phase_name == "P3_YELLOW":
        state.update({"S_R": False, "S_Y": True})
    return state


class TrafficLightMonitorUI:
    def __init__(self) -> None:
        rospy.init_node("traffic_light_monitor_ui", anonymous=True)
        if pygame is None:
            raise RuntimeError("pygame is required for tl_monitor_ui")

        self.width = int(rospy.get_param("~width", 1240))
        self.height = int(rospy.get_param("~height", 1020))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 30.0))
        self.panel_w = int(rospy.get_param("~panel_width", 360))
        self.panel_h = int(rospy.get_param("~panel_height", 230))
        self.panel_margin = int(rospy.get_param("~panel_margin", 16))
        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.tl_stop_distance_m = float(rospy.get_param("~tl_stop_distance_m", 15.0))
        self.map_height = int(rospy.get_param("~map_height", 720))
        self.show_map_view = bool(rospy.get_param("~show_map_view", True))

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
        self.GREY2 = (60, 60, 60)
        self.MAP_BG = (25, 25, 30)
        # Intersection layout (matches traffic_manager.py)
        self._intersection_layout = {
            "A": {"pos": (420, 130), "angle": 180, "label": "Intersection 1"},
            "B": {"pos": (80, 460), "angle": -90, "label": "Intersection 2"},
            "C": {"pos": (820, 460), "angle": 90, "label": "Intersection 3"},
        }

        rospy.Subscriber("/traffic_phase", TrafficLightPhase, self._phase_cb, queue_size=10)
        self._running = True

        # Subscribe vehicles (odometry + command)
        for idx in range(self.num_vehicles):
            role = f"ego_vehicle_{idx+1}"
            self._vehicles[role] = {"pos": None, "speed": None}
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role, queue_size=10)
            rospy.Subscriber(f"/carla/{role}/vehicle_control_cmd", AckermannDrive, self._cmd_cb, callback_args=role, queue_size=10)

    # ----- Signal rendering borrowed from traffic_manager.py -----
    def _draw_signal_box_4light(self, surface, x, y, title, show_label=True):
        pygame.draw.rect(surface, self.BLACK, (x, y, 100, 180), border_radius=8)
        if show_label and title:
            label = self.font.render(title, True, self.WHITE)
            surface.blit(label, (x, y - 18))

    def _draw_signal_box_3light(self, surface, x, y, title, show_label=True):
        pygame.draw.rect(surface, self.BLACK, (x, y, 100, 150), border_radius=8)
        if show_label and title:
            label = self.font.render(title, True, self.WHITE)
            surface.blit(label, (x, y - 18))

    def _draw_light(self, surface, x, y, color, on):
        radius = 16
        if on:
            pygame.draw.circle(surface, color, (x, y), radius)
        else:
            pygame.draw.circle(surface, self.GREY2, (x, y), radius)
            pygame.draw.circle(surface, color, (x, y), radius, 2)

    def _draw_arrow_light(self, surface, x, y, on):
        radius = 16
        if on:
            pygame.draw.circle(surface, self.GREEN, (x, y), radius)
        else:
            pygame.draw.circle(surface, self.GREY2, (x, y), radius)
            pygame.draw.circle(surface, self.GREEN, (x, y), radius, 2)
        arrow_text = self.font.render("<-", True, self.WHITE if on else self.GREEN)
        surface.blit(arrow_text, arrow_text.get_rect(center=(x, y)))

    def _draw_horizontal_signal(self, surface, x, y, title, state, show_label=True):
        pygame.draw.rect(surface, self.BLACK, (x, y, 150, 70), border_radius=8)
        if show_label and title:
            surface.blit(self.font.render(title, True, self.WHITE), (x, y - 18))
        positions = [
            (x + 35, y + 35, self.RED, state["S_R"]),
            (x + 75, y + 35, self.YELLOW, state["S_Y"]),
            (x + 115, y + 35, self.GREEN, state["S_G"]),
        ]
        for px, py, color, on in positions:
            self._draw_light(surface, px, py, color, on)

    def _draw_intersection_raw(self, surface, base_x, base_y, name, state, show_labels=True):
        x1, y1 = base_x, base_y
        self._draw_signal_box_4light(surface, x1, y1, f"{name} L->R", show_label=show_labels)
        self._draw_light(surface, x1 + 50, y1 + 30, self.RED, state["M_LR_R"])
        self._draw_light(surface, x1 + 50, y1 + 70, self.YELLOW, state["M_LR_Y"])
        self._draw_light(surface, x1 + 50, y1 + 110, self.GREEN, state["M_LR_G"])
        self._draw_arrow_light(surface, x1 + 50, y1 + 150, state["M_LR_LEFT"])

        x2, y2 = base_x + 150, base_y + 40
        self._draw_signal_box_3light(surface, x2, y2, f"{name} R->L", show_label=show_labels)
        self._draw_light(surface, x2 + 50, y2 + 30, self.RED, state["M_RL_R"])
        self._draw_light(surface, x2 + 50, y2 + 70, self.YELLOW, state["M_RL_Y"])
        self._draw_light(surface, x2 + 50, y2 + 110, self.GREEN, state["M_RL_G"])

        self._draw_horizontal_signal(surface, base_x + 25, base_y - 100, f"{name} Side", state, show_label=show_labels)

    def _draw_intersection(self, base_x, base_y, name, state, angle: int = 0):
        if angle == 0:
            self._draw_intersection_raw(self.screen, base_x, base_y, name, state, show_labels=True)
            anchor_x = base_x + 150
            anchor_y = base_y - 10
        else:
            temp_width, temp_height = 400, 400
            temp_surface = pygame.Surface((temp_width, temp_height), pygame.SRCALPHA)
            local_base_x, local_base_y = 80, 140
            self._draw_intersection_raw(temp_surface, local_base_x, local_base_y, name, state, show_labels=False)
            rotated = pygame.transform.rotate(temp_surface, angle)
            center_x = base_x + 150
            center_y = base_y + 120
            rect = rotated.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated, rect.topleft)
            anchor_x = rect.centerx
            anchor_y = rect.top - 10

        label = self.font.render(name, True, self.WHITE)
        label_rect = label.get_rect(center=(anchor_x, anchor_y))
        self.screen.blit(label, label_rect)

    def _draw_map_view(self, intersection_views: List[Tuple[str, Dict, List[Dict]]]) -> None:
        map_h = min(self.map_height, self.height - self.panel_margin - 10)
        pygame.draw.rect(self.screen, self.MAP_BG, (0, 0, self.width, map_h))
        for iid, cfg in self._intersection_layout.items():
            view = next((item for item in intersection_views if item[0] == iid), None)
            if view is None:
                continue
            _, info, approaches = view
            state = info.get("light_states") or get_light_states(info.get("phase", ""))
            label = cfg.get("label", iid)
            self._draw_intersection(cfg["pos"][0], cfg["pos"][1], label, state, cfg.get("angle", 0))
            phase_txt = info.get("phase", "")
            text = self.font.render(f"{iid} : {phase_txt}", True, self.WHITE)
            text_pos = (cfg["pos"][0], max(8, cfg["pos"][1] - 40))
            self.screen.blit(text, text_pos)
            info_y = cfg["pos"][1] + 190
            for ap in approaches[:3]:
                names = ", ".join(ap.get("vehicles", [])) or "-"
                ap_text = self.font.render(f"{ap.get('name','')}: {names}", True, self.CYAN)
                self.screen.blit(ap_text, (cfg["pos"][0], info_y))
                info_y += 16

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
                "light_states": get_light_states(msg.phase or ""),
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

    def _snapshot_vehicle_states(self) -> Dict[str, Dict]:
        with self._lock:
            return {
                role: {"pos": st.get("pos"), "speed": st.get("speed")}
                for role, st in self._vehicles.items()
            }

    def _vehicles_in_region(self, veh_states: Dict[str, Dict], rect: Tuple[float, float, float, float], stop: Tuple[float, float]) -> List[str]:
        xmin, xmax, ymin, ymax = rect
        sx, sy = stop
        hits: List[Tuple[float, str]] = []
        for role, st in veh_states.items():
            pos = st.get("pos")
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                dx = sx - x
                dy = sy - y
                dist = math.hypot(dx, dy)
                hits.append((dist, role))
        hits.sort(key=lambda item: item[0])
        return [role for _, role in hits]

    def _augment_approaches(self, approaches: List[Dict], veh_states: Dict[str, Dict]) -> List[Dict]:
        enriched: List[Dict] = []
        for ap in approaches:
            rect = ap.get("rect", (0.0, 0.0, 0.0, 0.0))
            stop = ap.get("stop", (0.0, 0.0))
            vehicles = self._vehicles_in_region(veh_states, rect, stop)
            enriched.append({
                "name": ap.get("name", ""),
                "color": ap.get("color", 0),
                "rect": rect,
                "stop": stop,
                "vehicles": vehicles,
            })
        return enriched

    def _draw_panel(self, x: int, y: int, w: int, h: int, title: str, phase: str, stamp: rospy.Time, approaches: List[Dict], veh_states: Dict) -> None:
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

        row_h = 40
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
            vehicles = ap.get("vehicles", [])
            veh_line = self.font.render(f"veh: {', '.join(vehicles) if vehicles else '-'}", True, self.CYAN)
            self.screen.blit(label, (ax + 36, ry + 4))
            self.screen.blit(meta, (ax + 36, ry + 18))
            self.screen.blit(veh_line, (ax + 36, ry + 32))

        # Vehicles section
        vy = ay + ah_top + 6
        pygame.draw.rect(self.screen, self.GREY, (ax, vy, aw, ah_bot), width=1)
        vtitle = self.font.render("vehicles in area", True, self.CYAN)
        self.screen.blit(vtitle, (ax + 6, vy + 4))
        v_row_h = 22
        v_list = self._collect_vehicles_for_intersection(approaches, veh_states)
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

    def _collect_vehicles_for_intersection(self, approaches: List[Dict], veh_states: Dict[str, Dict]) -> List[Dict]:
        """Return list of dicts: role,color,dist,speed,decision for vehicles inside these approaches."""
        out: List[Dict] = []
        # Build flat approach list with color and stopline
        aps: List[Tuple[Tuple[float, float, float, float], Tuple[float, float], int]] = []
        for ap in approaches:
            rect = ap.get("rect", (0.0, 0.0, 0.0, 0.0))
            stop = ap.get("stop", (0.0, 0.0))
            color = int(ap.get("color", 0))
            aps.append((rect, stop, color))
        for role, st in veh_states.items():
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
                inter_copy = dict(self._intersections)
            veh_snapshot = self._snapshot_vehicle_states()
            inter_views: List[Tuple[str, Dict, List[Dict]]] = []
            for iid, data in inter_copy.items():
                approaches = self._augment_approaches(data.get("approaches", []), veh_snapshot)
                inter_views.append((iid, data, approaches))
            panel_origin_y = self.panel_margin
            if self.show_map_view:
                self._draw_map_view(inter_views)
                panel_origin_y = self.map_height + self.panel_margin
            items = inter_views
            cols = max(1, self.width // (self.panel_w + self.panel_margin))
            for idx, (iid, data, approaches) in enumerate(items):
                col = idx % cols
                row = idx // cols
                px = self.panel_margin + col * (self.panel_w + self.panel_margin)
                py = panel_origin_y + row * (self.panel_h + self.panel_margin)
                if py + self.panel_h > self.height - self.panel_margin:
                    continue
                self._draw_panel(
                    px,
                    py,
                    self.panel_w,
                    self.panel_h,
                    f"{iid}",
                    str(data.get("phase", "")),
                    data.get("stamp", rospy.Time(0)),
                    approaches,
                    veh_snapshot,
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
