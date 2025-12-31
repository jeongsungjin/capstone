#!/usr/bin/env python3
"""
custom_paths의 좌표가 CARLA 맵 도로에 스냅되는지 빠르게 검사하는 스크립트.
- target edge: 파일명 custom_path_{u}to{v}.json
- 포맷: {"points": [[x, y], ...]} 또는 {"points": [{"x":..,"y":..}, ...]}
- 출력: 스냅 성공/실패 개수, 실패 샘플, entry/exit 근접도

usage:
  rosrun perception check_custom_paths.py _u:=21 _v:=24
"""

import json
import math
import os
from typing import List, Tuple

import rospy
import rospkg

try:
    import setup_carla_path  # noqa: F401
except Exception:
    # setup_carla_path가 없으면 계속 시도 (CARLA가 PYTHONPATH에 이미 있을 수 있음)
    pass

try:
    import carla
except Exception as exc:  # pragma: no cover
    carla = None
    # 노드 초기화 이전이라 rospy 사용이 어려워 print로 알림
    print(f"[check_custom_paths] Failed to import CARLA: {exc}")


def load_points(data) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if isinstance(data, dict) and "points" in data and isinstance(data["points"], list):
        data = data["points"]
    if not isinstance(data, list):
        return pts
    for pt in data:
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            pts.append((float(pt["x"]), float(pt["y"])))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((float(pt[0]), float(pt[1])))
    return pts


def main():
    rospy.init_node("check_custom_paths", anonymous=True)

    u = int(rospy.get_param("~u", -1))
    v = int(rospy.get_param("~v", -1))
    carla_host = rospy.get_param("~carla_host", "localhost")
    carla_port = int(rospy.get_param("~carla_port", 2000))
    town = rospy.get_param("~carla_town", None)

    if u < 0 or v < 0:
        rospy.logfatal("Set ~u and ~v (edge ids) to check.")
        return

    if carla is None:
        rospy.logfatal("CARLA API not available.")
        return

    client = carla.Client(carla_host, carla_port)
    client.set_timeout(5.0)
    world = client.get_world() if town is None else client.load_world(town)
    cmap = world.get_map()

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("perception")
    fname = os.path.join(pkg_path, "scripts", "custom_paths", f"custom_path_{u}to{v}.json")
    if not os.path.isfile(fname):
        rospy.logfatal("File not found: %s", fname)
        return

    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = load_points(data)
    if len(pts) < 2:
        rospy.logfatal("Points too short in %s", fname)
        return

    snapped = 0
    failed = 0
    failed_samples = []
    z_ref = 0.0
    for i, (x, y) in enumerate(pts):
        wp = None
        for z_try in (z_ref, 0.0):
            loc = carla.Location(x=x, y=y, z=z_try)
            try:
                wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any)
            except Exception:
                wp = None
            if wp is not None:
                snapped += 1
                z_ref = wp.transform.location.z
                break
        if wp is None:
            failed += 1
            if len(failed_samples) < 5:
                failed_samples.append((x, y))

    # entry/exit 근접도: 첫/마지막 포인트 기준
    # 그래프 entry/exit는 모를 수 있으므로 단순히 첫/끝 거리만 정보 표시
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    d_start_end = dist(pts[0], pts[-1])

    rospy.loginfo(
        "[check_custom_paths] %s snapped=%d failed=%d (total=%d) start=(%.2f,%.2f) end=(%.2f,%.2f) d_start_end=%.2fm failed_samples=%s",
        fname,
        snapped,
        failed,
        len(pts),
        pts[0][0],
        pts[0][1],
        pts[-1][0],
        pts[-1][1],
        d_start_end,
        failed_samples,
    )


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


