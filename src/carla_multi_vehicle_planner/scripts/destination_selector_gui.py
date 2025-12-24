#!/usr/bin/env python3
"""
GUI 기반 목적지 선택기
- 차량 선택 버튼: 1~5
- 목적지 선택 버튼: 1~7 (YAML regions center를 도로로 snap)
- 선택 시 /override_goal/<role> 로 PoseStamped 퍼블리시

참고: 기존 터미널 버전(destination_selector.py)을 GUI로 대체/보완
"""

import tkinter as tk
from functools import partial
from typing import Dict

import rospy
import yaml
from geometry_msgs.msg import PoseStamped

try:
    import carla  # type: ignore
except Exception as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


class DestinationSelectorGUI:
    def __init__(self) -> None:
        rospy.init_node("destination_selector_gui", anonymous=True, disable_signals=True)
        if carla is None:
            raise RuntimeError("CARLA API unavailable")

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 5))
        self.map_yaml = rospy.get_param("~map_yaml", "carla_regions_map_package_Maps_ces_ces_20251224_011231.yaml")
        self.carla_host = rospy.get_param("~carla_host", "localhost")
        self.carla_port = int(rospy.get_param("~carla_port", 2000))
        self.default_z = float(rospy.get_param("~default_z", 0.1))

        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        self.destinations: Dict[int, carla.Location] = self._load_destinations()
        self.publishers: Dict[str, rospy.Publisher] = {}
        self.current_role = "ego_vehicle_1"

        self.root = tk.Tk()
        self.root.title("Destination Selector")
        self._build_ui()

    def _load_destinations(self) -> Dict[int, carla.Location]:
        dests: Dict[int, carla.Location] = {}
        try:
            with open(self.map_yaml, "r") as f:
                data = yaml.safe_load(f)
            regions = data.get("regions", []) if isinstance(data, dict) else []
        except Exception as exc:
            rospy.logwarn(f"dest_selector_gui: failed to load yaml {self.map_yaml}: {exc}")
            regions = []
        for idx, reg in enumerate(regions, start=1):
            if idx > 7:
                break
            center = reg.get("center")
            if not center or len(center) < 2:
                continue
            cx, cy = float(center[0]), float(center[1])
            loc = carla.Location(x=cx, y=cy, z=self.default_z)
            try:
                wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is not None:
                    loc = wp.transform.location
            except Exception:
                pass
            dests[idx] = loc
            rospy.loginfo(f"[GUI] dest {idx}: snapped to ({loc.x:.2f},{loc.y:.2f})")
        return dests

    def _build_ui(self) -> None:
        frame_roles = tk.LabelFrame(self.root, text="Vehicle")
        frame_roles.pack(fill="x", padx=8, pady=4)
        for i in range(1, min(5, self.num_vehicles) + 1):
            btn = tk.Button(frame_roles, text=str(i), width=6, command=partial(self._select_role, i))
            btn.pack(side="left", padx=2, pady=2)

        frame_dest = tk.LabelFrame(self.root, text="Destination")
        frame_dest.pack(fill="x", padx=8, pady=4)
        for i in range(1, 8):
            btn = tk.Button(frame_dest, text=f"{i}", width=6, command=partial(self._send_dest, i))
            btn.pack(side="left", padx=2, pady=2)

        self.label_status = tk.Label(self.root, text="role: ego_vehicle_1  dest: -")
        self.label_status.pack(fill="x", padx=8, pady=8)

    def _select_role(self, idx: int) -> None:
        self.current_role = f"ego_vehicle_{idx}"
        self._update_status(dest_id=None)

    def _send_dest(self, dest_id: int) -> None:
        loc = self.destinations.get(dest_id)
        if loc is None:
            rospy.logwarn(f"dest {dest_id} not loaded")
            return
        topic = f"/override_goal/{self.current_role}"
        if topic not in self.publishers:
            self.publishers[topic] = rospy.Publisher(topic, PoseStamped, queue_size=1)
            rospy.sleep(0.05)
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = loc.x
        msg.pose.position.y = loc.y
        msg.pose.position.z = loc.z
        msg.pose.orientation.w = 1.0
        self.publishers[topic].publish(msg)
        rospy.loginfo(f"[GUI] {self.current_role} -> dest {dest_id} ({loc.x:.2f},{loc.y:.2f})")
        self._update_status(dest_id=dest_id)

    def _update_status(self, dest_id=None):
        txt = f"role: {self.current_role}"
        if dest_id is not None:
            txt += f"  dest: {dest_id}"
        self.label_status.config(text=txt)

    def spin(self) -> None:
        # rospy spin in background to keep publishers alive
        def _spin():
            rospy.spin()
        import threading

        th = threading.Thread(target=_spin, daemon=True)
        th.start()
        self.root.mainloop()


if __name__ == "__main__":
    try:
        node = DestinationSelectorGUI()
        node.spin()
    except Exception as exc:
        rospy.logfatal(f"destination_selector_gui: {exc}")

