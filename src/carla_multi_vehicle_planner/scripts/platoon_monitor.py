#!/usr/bin/env python3

import math
import sys
import threading
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDrive

try:
    from python_qt_binding import QtCore, QtGui, QtWidgets  # type: ignore
except Exception as exc:
    rospy.logfatal("platoon_monitor: failed to import python_qt_binding (Qt). %s", exc)
    raise


def _quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _project_point_to_path(points: List[Tuple[float, float]], px: float, py: float) -> Optional[float]:
    if not points or len(points) < 2:
        return None
    best = None
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            continue
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        proj_x = x1 + dx * t
        proj_y = y1 + dy * t
        dist = math.hypot(px - proj_x, py - proj_y)
        if best is None or dist < best:
            best = dist
    return best


class RoleStatus:
    def __init__(self) -> None:
        self.tracking_ok: bool = False
        self.last_tracking_stamp: float = 0.0
        self.tracking_pose: Optional[PoseStamped] = None
        self.odom: Optional[Odometry] = None
        self.path_points: List[Tuple[float, float]] = []
        self.path_stamp: float = 0.0
        self.manual_stop_active: bool = False
        self.last_manual_toggle: float = 0.0
        self.last_stop_publish: float = 0.0


class PlatoonMonitorWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Platoon / BEV Monitoring")
        self.resize(960, 360)

        self.num_vehicles = int(rospy.get_param("~num_vehicles", 3))
        self.roles = [f"ego_vehicle_{i+1}" for i in range(self.num_vehicles)]
        self.enable_platooning = bool(rospy.get_param("~enable_platooning", False))
        self.enable_bev_pipeline = bool(rospy.get_param("~enable_bev_pipeline", False))
        self.enable_bev_tracking_hold = bool(rospy.get_param("~enable_bev_tracking_hold", False))
        self.platoon_leader = str(rospy.get_param("~platoon_leader", "ego_vehicle_1"))
        followers_str = str(rospy.get_param("~platoon_followers", "ego_vehicle_2,ego_vehicle_3")).strip()
        self.platoon_followers = [s.strip() for s in followers_str.split(",") if s.strip()]

        self.status: Dict[str, RoleStatus] = {role: RoleStatus() for role in self.roles}
        self._lock = threading.Lock()
        self.stop_publishers: Dict[str, rospy.Publisher] = {
            role: rospy.Publisher(
                f"/carla/{role}/vehicle_control_cmd_override", AckermannDrive, queue_size=1
            )
            for role in self.roles
        }
        self._shortcuts: List[QtWidgets.QShortcut] = []
        self.manual_stop_interval = float(rospy.get_param("~manual_stop_interval", 0.2))

        self._build_ui()
        self._setup_subscribers()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_table)
        self._timer.start(500)
        self._install_shortcuts()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.info_label = QtWidgets.QLabel(self)
        layout.addWidget(self.info_label)

        columns = [
            "Role",
            "Platoon",
            "Tracking",
            "Position (x,y)",
            "Heading (deg)",
            "Match Err (m)",
            "Path Dev (m)",
            "Status",
        ]
        self.table = QtWidgets.QTableWidget(len(self.roles), len(columns), self)
        self.table.setHorizontalHeaderLabels(columns)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)

    def _setup_subscribers(self) -> None:
        for role in self.roles:
            rospy.Subscriber(
                f"/carla/{role}/bev_tracking_ok",
                Bool,
                lambda msg, r=role: self._tracking_cb(msg, r),
                queue_size=1,
            )
            rospy.Subscriber(
                f"/carla/{role}/odometry",
                Odometry,
                lambda msg, r=role: self._odom_cb(msg, r),
                queue_size=1,
            )
            rospy.Subscriber(
                f"/planned_path_{role}",
                Path,
                lambda msg, r=role: self._path_cb(msg, r),
                queue_size=1,
            )
            rospy.Subscriber(
                f"/bev_tracking_pose/{role}",
                PoseStamped,
                lambda msg, r=role: self._bev_pose_cb(msg, r),
                queue_size=1,
            )

    def _install_shortcuts(self) -> None:
        for idx, role in enumerate(self.roles, start=1):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(str(idx)), self)
            shortcut.setContext(QtCore.Qt.WindowShortcut)
            shortcut.activated.connect(lambda r=role: self._toggle_manual_stop(r))
            self._shortcuts.append(shortcut)

    def _toggle_manual_stop(self, role: str) -> None:
        with self._lock:
            status = self.status.get(role)
            if not status:
                return
            status.manual_stop_active = not status.manual_stop_active
            status.last_manual_toggle = rospy.Time.now().to_sec()
            active = status.manual_stop_active
        if active:
            self._publish_stop_command(role)
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                f"{role}: STOP engaged",
                self,
                self.rect(),
                1500,
            )
        else:
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                f"{role}: STOP released",
                self,
                self.rect(),
                1500,
            )

    def _publish_stop_command(self, role: str) -> None:
        pub = self.stop_publishers.get(role)
        if pub is None:
            return
        cmd = AckermannDrive()
        cmd.speed = 0.0
        cmd.steering_angle = 0.0
        try:
            pub.publish(cmd)
        except Exception:
            return
        with self._lock:
            status = self.status.get(role)
            if status:
                status.last_stop_publish = rospy.Time.now().to_sec()

    def _tracking_cb(self, msg: Bool, role: str) -> None:
        with self._lock:
            status = self.status.get(role)
            if status:
                status.tracking_ok = bool(msg.data)
                status.last_tracking_stamp = rospy.Time.now().to_sec()

    def _bev_pose_cb(self, msg: PoseStamped, role: str) -> None:
        with self._lock:
            status = self.status.get(role)
            if status:
                status.tracking_pose = msg

    def _odom_cb(self, msg: Odometry, role: str) -> None:
        with self._lock:
            status = self.status.get(role)
            if status:
                status.odom = msg

    def _path_cb(self, msg: Path, role: str) -> None:
        if len(msg.poses) < 2:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            status = self.status.get(role)
            if status:
                status.path_points = pts
                status.path_stamp = rospy.Time.now().to_sec()

    def _role_platoon_label(self, role: str) -> str:
        if not self.enable_platooning:
            return "Solo"
        if role == self.platoon_leader:
            return "Leader"
        if role in self.platoon_followers:
            return "Follower"
        return "Solo"

    def _compute_match_error(self, status: RoleStatus) -> Optional[float]:
        if status.odom is None or status.tracking_pose is None:
            return None
        ox = status.odom.pose.pose.position.x
        oy = status.odom.pose.pose.position.y
        tx = status.tracking_pose.pose.position.x
        ty = status.tracking_pose.pose.position.y
        return math.hypot(ox - tx, oy - ty)

    def _compute_path_error(self, status: RoleStatus) -> Optional[float]:
        if status.odom is None or not status.path_points:
            return None
        ox = status.odom.pose.pose.position.x
        oy = status.odom.pose.pose.position.y
        return _project_point_to_path(status.path_points, ox, oy)

    def _set_cell(self, row: int, col: int, text: str, color: Optional[QtGui.QColor] = None) -> None:
        item = QtWidgets.QTableWidgetItem(text)
        if color:
            item.setBackground(color)
        self.table.setItem(row, col, item)

    def _update_table(self) -> None:
        now_sec = rospy.Time.now().to_sec()
        info_text = f"BEV pipeline: {'ON' if self.enable_bev_pipeline else 'OFF'}"
        if self.enable_platooning:
            info_text += f" | Platooning: ON (Leader: {self.platoon_leader}, Followers: {', '.join(self.platoon_followers)})"
        else:
            info_text += " | Platooning: OFF"
        info_text += f" | Tracking Hold: {'ON' if self.enable_bev_tracking_hold else 'OFF'}"
        info_text += " | Keys 1-9: toggle stop per role"
        self.info_label.setText(info_text)

        for row, role in enumerate(self.roles):
            with self._lock:
                status = self.status.get(role)
                tracking_ok = status.tracking_ok if status else False
                odom = status.odom if status else None
                match_err = self._compute_match_error(status) if status else None
                path_err = self._compute_path_error(status) if status else None

            self._set_cell(row, 0, role)
            self._set_cell(row, 1, self._role_platoon_label(role))

            track_color = QtGui.QColor(46, 204, 113) if tracking_ok else QtGui.QColor(231, 76, 60)
            track_text = "OK" if tracking_ok else "LOST"
            self._set_cell(row, 2, track_text, track_color)

            if odom:
                ox = odom.pose.pose.position.x
                oy = odom.pose.pose.position.y
                yaw = _quaternion_to_yaw(odom.pose.pose.orientation)
                self._set_cell(row, 3, f"{ox:.1f}, {oy:.1f}")
                self._set_cell(row, 4, f"{math.degrees(yaw):.1f}")
            else:
                self._set_cell(row, 3, "N/A")
                self._set_cell(row, 4, "N/A")

            if match_err is not None:
                match_color = QtGui.QColor(46, 204, 113) if match_err < 0.5 else QtGui.QColor(243, 156, 18)
                self._set_cell(row, 5, f"{match_err:.2f}", match_color)
            else:
                self._set_cell(row, 5, "N/A")

            if path_err is not None:
                path_color = QtGui.QColor(46, 204, 113) if path_err < 0.5 else QtGui.QColor(241, 196, 15)
                self._set_cell(row, 6, f"{path_err:.2f}", path_color)
            else:
                self._set_cell(row, 6, "N/A")

            status_text = []
            status_color = QtGui.QColor(46, 204, 113)
            if not tracking_ok:
                status_text.append("Tracking Hold")
                status_color = QtGui.QColor(231, 76, 60)
            if match_err is not None and match_err >= 1.0:
                status_text.append("Match drift")
                status_color = QtGui.QColor(230, 126, 34)
            if path_err is not None and path_err >= 1.0:
                status_text.append("Off-path")
                status_color = QtGui.QColor(230, 126, 34)
            if status and status.manual_stop_active:
                status_text.append("Manual STOP")
                status_color = QtGui.QColor(52, 152, 219)
                if (now_sec - status.last_stop_publish) >= self.manual_stop_interval:
                    self._publish_stop_command(role)
            summary = ", ".join(status_text) if status_text else "Nominal"
            self._set_cell(row, 7, summary, status_color)



def main() -> None:
    rospy.init_node("platoon_monitor", anonymous=False, disable_signals=True)
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True

    widget = PlatoonMonitorWidget()
    widget.show()

    def _on_shutdown() -> None:
        if widget.isVisible():
            widget.close()
        if created_app:
            app.quit()

    rospy.on_shutdown(_on_shutdown)

    # Keep Qt event loop alive; rospy callbacks run in background threads.
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    try:
        app.exec_()
    except Exception:
        pass


if __name__ == "__main__":
    main()

