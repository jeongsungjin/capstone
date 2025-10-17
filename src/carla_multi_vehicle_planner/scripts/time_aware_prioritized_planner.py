#!/usr/bin/env python3
"""Time-aware prioritized planning utilities for CARLA waypoint routes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - CARLA not available in unit tests
    carla = None

# Optional ROS logging (used for debug traces when available)
try:  # pragma: no cover - optional dependency in pure unit tests
    import rospy  # type: ignore
except Exception:  # pragma: no cover
    rospy = None


@dataclass
class ScheduledVisit:
    """Represents the arrival and departure of an agent at a node."""

    node_key: Tuple[int, int]
    position: Tuple[float, float]
    arrival: int  # inclusive start timestep
    departure: int  # exclusive end timestep, includes buffer and waits

    def wait_steps(self) -> int:
        return max(0, self.departure - self.arrival)


class ReservationTable:
    """Discretised spatiotemporal reservation table."""

    def __init__(self):
        self._node: Dict[Tuple[Tuple[int, int], int], str] = {}
        self._edge: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], str] = {}

    def is_node_free(
        self,
        node_key: Tuple[int, int],
        start: int,
        end: int,
        ignore: Optional[str] = None,
    ) -> bool:
        for step in range(start, end):
            occupant = self._node.get((node_key, step))
            if occupant is not None and occupant != ignore:
                return False
        return True

    def is_edge_free(
        self,
        edge_key: Tuple[Tuple[int, int], Tuple[int, int]],
        start: int,
        end: int,
        ignore: Optional[str] = None,
    ) -> bool:
        if start >= end:
            return True
        a, b = edge_key
        for step in range(start, end):
            occupant = self._edge.get((a, b, step))
            if occupant is not None and occupant != ignore:
                return False
            occupant_rev = self._edge.get((b, a, step))
            if occupant_rev is not None and occupant_rev != ignore:
                return False
        return True

    def reserve_node(self, agent_id: str, node_key: Tuple[int, int], start: int, end: int) -> None:
        for step in range(start, end):
            self._node[(node_key, step)] = agent_id

    def reserve_edge(
        self,
        agent_id: str,
        edge_key: Tuple[Tuple[int, int], Tuple[int, int]],
        start: int,
        end: int,
    ) -> None:
        if start >= end:
            return
        a, b = edge_key
        for step in range(start, end):
            self._edge[(a, b, step)] = agent_id

    def release_agent(self, agent_id: str) -> None:
        node_keys = [key for key, occupant in self._node.items() if occupant == agent_id]
        for key in node_keys:
            self._node.pop(key, None)
        edge_keys = [key for key, occupant in self._edge.items() if occupant == agent_id]
        for key in edge_keys:
            self._edge.pop(key, None)

    def clear(self) -> None:
        self._node.clear()
        self._edge.clear()


@dataclass
class ScheduleResult:
    success: bool
    visits: Optional[List[ScheduledVisit]] = None
    expanded_path: Optional[List[Tuple[float, float]]] = None
    reason: Optional[str] = None


class TimeAwarePrioritizedPlanner:
    """Builds time-feasible schedules on top of waypoint routes."""

    def __init__(
        self,
        dt: float = 0.2,
        nominal_speed: float = 10.0,
        buffer_time: float = 0.4,
        max_wait_time: float = 5.0,
        reservation_horizon: float = 40.0,
        proximity_resolution: float = 0.5,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nominal_speed <= 0.0:
            raise ValueError("nominal_speed must be positive")
        self.dt = dt
        self.nominal_speed = nominal_speed
        self.buffer_steps = max(1, int(math.ceil(buffer_time / dt)))
        self.max_wait_steps = max(0, int(math.ceil(max_wait_time / dt)))
        self.horizon_steps = (
            max(0, int(math.ceil(reservation_horizon / dt))) if reservation_horizon > 0.0 else 0
        )
        self.key_resolution = max(1.0, proximity_resolution)
        self._table = ReservationTable()
        self._dist_per_step = self.nominal_speed * self.dt

        # Debug switch read from ROS param if available
        self._debug_enabled = False
        if rospy is not None:
            try:
                self._debug_enabled = bool(rospy.get_param("/networkx_path_planner/schedule_debug", False))
            except Exception:
                self._debug_enabled = False

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled:
            if rospy is not None:
                rospy.logdebug(msg)
            else:
                print(msg)

    @property
    def table(self) -> ReservationTable:
        return self._table

    def node_key(self, waypoint: carla.Waypoint) -> Tuple[int, int]:  # type: ignore[valid-type]
        loc = waypoint.transform.location
        return (
            int(round(loc.x / self.key_resolution)),
            int(round(loc.y / self.key_resolution)),
        )

    @staticmethod
    def _location_tuple(waypoint: carla.Waypoint) -> Tuple[float, float]:  # type: ignore[valid-type]
        loc = waypoint.transform.location
        return (loc.x, loc.y)

    def release_agent(self, agent_id: str) -> None:
        self._table.release_agent(agent_id)

    def clear(self) -> None:
        self._table.clear()

    def schedule_route(
        self,
        agent_id: str,
        route_waypoints: Sequence[carla.Waypoint],  # type: ignore[valid-type]
        base_path_points: Optional[Sequence[Tuple[float, float]]] = None,
        start_time_step: int = 0,
    ) -> ScheduleResult:
        if carla is None:
            return ScheduleResult(False, reason="CARLA API unavailable")
        if not route_waypoints:
            return ScheduleResult(False, reason="empty route")

        self._dbg(
            (
                f"[sched] agent={agent_id} waypoints={len(route_waypoints)} "
                f"dt={self.dt:.3f}s v={self.nominal_speed:.1f}m/s buffer_steps={self.buffer_steps} "
                f"max_wait_steps={self.max_wait_steps} horizon_steps={self.horizon_steps} "
                f"key_res={self.key_resolution:.2f}"
            )
        )

        # Build node sequence and ensure uniqueness across consecutive duplicates.
        node_sequence: List[Tuple[Tuple[int, int], carla.Waypoint]] = []
        for waypoint in route_waypoints:
            key = self.node_key(waypoint)
            if node_sequence and node_sequence[-1][0] == key:
                continue
            node_sequence.append((key, waypoint))
        if len(node_sequence) < 2:
            return ScheduleResult(False, reason="route too short for scheduling")

        visits: List[ScheduledVisit] = []

        start_key, start_wp = node_sequence[0]
        arrival = max(0, start_time_step)
        wait_extensions = 0
        while not self._table.is_node_free(start_key, arrival, arrival + self.buffer_steps):
            arrival += 1
            wait_extensions += 1
            if wait_extensions > self.max_wait_steps:
                return ScheduleResult(False, reason="start position congestion")
        departure = arrival + self.buffer_steps
        self._table.reserve_node(agent_id, start_key, arrival, departure)
        visits.append(
            ScheduledVisit(
                node_key=start_key,
                position=self._location_tuple(start_wp),
                arrival=arrival,
                departure=departure,
            )
        )

        total_wait_steps = wait_extensions
        if total_wait_steps > 0:
            self._dbg(f"[sched] agent={agent_id} start wait_extensions={total_wait_steps} steps")

        for idx in range(1, len(node_sequence)):
            prev_visit = visits[-1]
            prev_key, prev_wp = node_sequence[idx - 1]
            next_key, next_wp = node_sequence[idx]

            segment = prev_wp.transform.location.distance(next_wp.transform.location)
            travel_steps = max(1, int(math.ceil(segment / self._dist_per_step)))

            departure = prev_visit.departure
            arrival_candidate = departure + travel_steps
            while True:
                if self.horizon_steps and arrival_candidate > self.horizon_steps:
                    return ScheduleResult(False, reason="reservation horizon exceeded")
                node_free = self._table.is_node_free(
                    next_key, arrival_candidate, arrival_candidate + self.buffer_steps, ignore=agent_id
                )
                edge_free = self._table.is_edge_free(
                    (prev_key, next_key), departure, arrival_candidate, ignore=agent_id
                )
                if node_free and edge_free:
                    break
                departure += 1
                arrival_candidate += 1
                total_wait_steps += 1
                if total_wait_steps > self.max_wait_steps:
                    return ScheduleResult(False, reason="wait budget exhausted")
                self._table.reserve_node(agent_id, prev_key, prev_visit.departure, departure)
                prev_visit.departure = departure
            self._dbg(
                (
                    f"[sched] agent={agent_id} seg={idx} depart={departure} arrive={arrival_candidate} "
                    f"wait_total={total_wait_steps}"
                )
            )

            self._table.reserve_edge(agent_id, (prev_key, next_key), departure, arrival_candidate)
            self._table.reserve_node(
                agent_id, next_key, arrival_candidate, arrival_candidate + self.buffer_steps
            )
            visits.append(
                ScheduledVisit(
                    node_key=next_key,
                    position=self._location_tuple(next_wp),
                    arrival=arrival_candidate,
                    departure=arrival_candidate + self.buffer_steps,
                )
            )

        expanded_path = self._expand_path_with_schedule(visits, base_path_points)
        self._dbg(
            (
                f"[sched] agent={agent_id} success visits={len(visits)} "
                f"total_wait_steps={total_wait_steps} horizon_s={visits[-1].departure * self.dt:.1f}"
            )
        )
        return ScheduleResult(True, visits=visits, expanded_path=expanded_path)

    def _expand_path_with_schedule(
        self,
        visits: Sequence[ScheduledVisit],
        base_points: Optional[Sequence[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        """Return geometry enriched with explicit wait segments for debugging."""

        if not visits:
            return []
        if base_points:
            # Fall back to base geometry – do not distort the original path.
            # (We still insert hold points at the start to visualise waiting.)
            enriched: List[Tuple[float, float]] = []
            first = visits[0]
            hold_points = max(0, first.departure - first.arrival)
            enriched.extend([first.position] * max(1, hold_points))
            for point in base_points:
                if not enriched or (abs(point[0] - enriched[-1][0]) > 1e-3 or abs(point[1] - enriched[-1][1]) > 1e-3):
                    enriched.append(point)
            return enriched

        # Without provided geometry, build a simple linear interpolation.
        path: List[Tuple[float, float]] = []
        for current_visit, next_visit in zip(visits[:-1], visits[1:]):
            if not path:
                path.append(current_visit.position)
            wait_steps = max(0, current_visit.departure - current_visit.arrival)
            for _ in range(wait_steps):
                path.append(current_visit.position)
            travel_steps = max(1, next_visit.arrival - current_visit.departure)
            for step in range(1, travel_steps + 1):
                ratio = step / float(travel_steps)
                interp = (
                    current_visit.position[0]
                    + (next_visit.position[0] - current_visit.position[0]) * ratio,
                    current_visit.position[1]
                    + (next_visit.position[1] - current_visit.position[1]) * ratio,
                )
                path.append(interp)
        path.append(visits[-1].position)
        return path


__all__ = [
    "ReservationTable",
    "ScheduledVisit",
    "ScheduleResult",
    "TimeAwarePrioritizedPlanner",
]
