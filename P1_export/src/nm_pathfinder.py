from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple
Box = Tuple[int, int, int, int]
Point = Tuple[float, float]
XY = Tuple[float, float]


def _point_in_box(p: Point, b: Box) -> bool:
    x, y = p
    x1, x2, y1, y2 = b
    return x1 <= x < x2 and y1 <= y < y2


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _clamp_point_to_box(p: Point, b: Box) -> Point:
    """Return the closest point to p that lies inside b."""
    x, y = p
    x1, x2, y1, y2 = b

    cx = _clamp(x, float(x1), float(x2 - 1))
    cy = _clamp(y, float(y1), float(y2 - 1))
    return (cx, cy)


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _find_containing_box(p: Point, boxes: Iterable[Box]) -> Optional[Box]:
    for b in boxes:
        if _point_in_box(p, b):
            return b
    return None


def _box_center_xy(b: Box) -> XY:
    x1, x2, y1, y2 = b
    return ((y1 + y2) / 2.0, (x1 + x2) / 2.0)


def _tri_area2(a: XY, b: XY, c: XY) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _shared_portal_xy(a: Box, b: Box) -> Tuple[XY, XY]:
    """
    Return the shared boundary segment ("portal") between two adjacent boxes as two endpoints in (x=col,y=row).
    """
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b

    if ax2 == bx1 or bx2 == ax1:
        shared_row = float(ax2 if ax2 == bx1 else bx2)
        lo_col = float(max(ay1, by1))
        hi_col = float(min(ay2, by2))
        if hi_col < lo_col:
            raise ValueError("Adjacent boxes have no overlapping portal (horizontal).")
        return (lo_col, shared_row), (hi_col, shared_row)

    if ay2 == by1 or by2 == ay1:
        shared_col = float(ay2 if ay2 == by1 else by2)
        lo_row = float(max(ax1, bx1))
        hi_row = float(min(ax2, bx2))
        if hi_row < lo_row:
            raise ValueError("Adjacent boxes have no overlapping portal (vertical).")
        return (shared_col, lo_row), (shared_col, hi_row)

    raise ValueError("Boxes are not edge-adjacent; no portal exists.")


def _orient_portal_lr(portal: Tuple[XY, XY], from_box: Box, to_box: Box) -> Tuple[XY, XY]:
    """
    Order portal endpoints into (left,right) relative to motion direction from from_box -> to_box.
    """
    p0, p1 = portal
    fx, fy = _box_center_xy(from_box)
    tx, ty = _box_center_xy(to_box)
    dx, dy = (tx - fx), (ty - fy)
    if dx == 0 and dy == 0:
        return (p0, p1) if p0 <= p1 else (p1, p0)

    v0 = (p0[0] - fx, p0[1] - fy)
    v1 = (p1[0] - fx, p1[1] - fy)
    c0 = dx * v0[1] - dy * v0[0]
    c1 = dx * v1[1] - dy * v1[0]

    if c0 == c1:
        return (p0, p1) if p0 <= p1 else (p1, p0)
    return (p0, p1) if c0 < c1 else (p1, p0)


def _string_pull(portals_lr: List[Tuple[XY, XY]]) -> List[XY]:
    """
    Funnel algorithm ("string pulling") over a sequence of portals.
    portals_lr includes degenerate start and end portals: (start,start) ... (end,end).
    """
    if not portals_lr:
        return []
    if len(portals_lr) == 1:
        return [portals_lr[0][0]]

    portal_apex = portals_lr[0][0]
    portal_left = portals_lr[0][0]
    portal_right = portals_lr[0][1]
    apex_index = 0
    left_index = 0
    right_index = 0

    path: List[XY] = [portal_apex]
    i = 1
    while i < len(portals_lr):
        left, right = portals_lr[i]

        if _tri_area2(portal_apex, portal_right, right) <= 0.0:
            if portal_apex == portal_right or _tri_area2(portal_apex, portal_left, right) > 0.0:
                portal_right = right
                right_index = i
            else:
                path.append(portal_left)
                portal_apex = portal_left
                apex_index = left_index
                portal_left = portal_apex
                portal_right = portal_apex
                left_index = apex_index
                right_index = apex_index
                i = apex_index + 1
                continue

        if _tri_area2(portal_apex, portal_left, left) >= 0.0:
            if portal_apex == portal_left or _tri_area2(portal_apex, portal_right, left) < 0.0:
                portal_left = left
                left_index = i
            else:
                path.append(portal_right)
                portal_apex = portal_right
                apex_index = right_index
                portal_left = portal_apex
                portal_right = portal_apex
                left_index = apex_index
                right_index = apex_index
                i = apex_index + 1
                continue

        i += 1

    end_pt = portals_lr[-1][0]
    if not path or path[-1] != end_pt:
        path.append(end_pt)
    return path


def find_path(source_point: Point, destination_point: Point, mesh) -> Tuple[List[Point], List[Box]]:
    """
    Search for a path from source_point to destination_point through a navmesh.

    Implements a **bidirectional A*** search over the navmesh boxes.

    Args:
        source_point: starting point (row, col) in the original image coordinate space
        destination_point: goal point (row, col) in the original image coordinate space
        mesh: dict with keys {'boxes', 'adj'} produced by nm_meshbuilder.py

    Returns:
        path: list of (row, col) points representing a polyline from source to destination
        explored: list of boxes expanded by the search
    """
    boxes: List[Box] = list(mesh.get("boxes", []))
    adj: Dict[Box, List[Box]] = dict(mesh.get("adj", {}))

    start_box = _find_containing_box(source_point, boxes)
    goal_box = _find_containing_box(destination_point, boxes)
    if start_box is None or goal_box is None:
        if start_box is None and goal_box is None:
            print("No path: source and destination are not inside any navmesh box.")
        elif start_box is None:
            print("No path: source point is not inside any navmesh box.")
        else:
            print("No path: destination point is not inside any navmesh box.")
        return [], []

    explored: List[Box] = []
    explored_set: Set[Box] = set()

    def _record(box: Box) -> None:
        """Record a box as 'visited' exactly once (stable order)."""
        if box not in explored_set:
            explored_set.add(box)
            explored.append(box)

    if start_box == goal_box:
        _record(start_box)
        return [
            (float(source_point[0]), float(source_point[1])),
            (float(destination_point[0]), float(destination_point[1])),
        ], explored

    _record(start_box)
    _record(goal_box)

    f_frontier: List[Tuple[float, float, Box]] = []
    b_frontier: List[Tuple[float, float, Box]] = []

    f_came_from: Dict[Box, Optional[Box]] = {start_box: None}
    b_came_from: Dict[Box, Optional[Box]] = {goal_box: None}

    f_cost: Dict[Box, float] = {start_box: 0.0}
    b_cost: Dict[Box, float] = {goal_box: 0.0}

    f_detail: Dict[Box, Point] = {start_box: (float(source_point[0]), float(source_point[1]))}
    b_detail: Dict[Box, Point] = {goal_box: (float(destination_point[0]), float(destination_point[1]))}

    heapq.heappush(f_frontier, (_euclid(source_point, destination_point), 0.0, start_box))
    heapq.heappush(b_frontier, (_euclid(destination_point, source_point), 0.0, goal_box))

    f_closed: Set[Box] = set()
    b_closed: Set[Box] = set()

    best_meet: Optional[Box] = None
    best_cost: float = float("inf")

    def _reconstruct_box_path(meet: Box) -> List[Box]:
        left: List[Box] = []
        cur: Optional[Box] = meet
        while cur is not None:
            left.append(cur)
            cur = f_came_from.get(cur)
        left.reverse()

        right: List[Box] = []
        cur = b_came_from.get(meet)
        while cur is not None:
            right.append(cur)
            cur = b_came_from.get(cur)

        return left + right

    def _reconstruct_point_path(box_path: List[Box]) -> List[Point]:
        start_xy: XY = (float(source_point[1]), float(source_point[0]))
        end_xy: XY = (float(destination_point[1]), float(destination_point[0]))

        portals_lr: List[Tuple[XY, XY]] = [(start_xy, start_xy)]
        for i in range(len(box_path) - 1):
            a, b = box_path[i], box_path[i + 1]
            portal = _shared_portal_xy(a, b)
            left, right = _orient_portal_lr(portal, a, b)
            portals_lr.append((left, right))
        portals_lr.append((end_xy, end_xy))

        pulled_xy = _string_pull(portals_lr)
        return [(p[1], p[0]) for p in pulled_xy]

    def _expand_forward() -> None:
        nonlocal best_meet, best_cost
        while f_frontier:
            _, g, current = heapq.heappop(f_frontier)
            if current in f_closed:
                continue
            f_closed.add(current)
            _record(current)

            if current in b_cost:
                cand = f_cost[current] + b_cost[current]
                if cand < best_cost:
                    best_cost = cand
                    best_meet = current

            current_point = f_detail[current]
            for nb in adj.get(current, []):
                nb_point = _clamp_point_to_box(current_point, nb)
                new_cost = f_cost[current] + _euclid(current_point, nb_point)
                if nb not in f_cost or new_cost < f_cost[nb]:
                    f_cost[nb] = new_cost
                    f_came_from[nb] = current
                    f_detail[nb] = nb_point
                    priority = new_cost + _euclid(nb_point, destination_point)
                    heapq.heappush(f_frontier, (priority, new_cost, nb))
            return

    def _expand_backward() -> None:
        nonlocal best_meet, best_cost
        while b_frontier:
            _, g, current = heapq.heappop(b_frontier)
            if current in b_closed:
                continue
            b_closed.add(current)
            _record(current)

            if current in f_cost:
                cand = b_cost[current] + f_cost[current]
                if cand < best_cost:
                    best_cost = cand
                    best_meet = current

            current_point = b_detail[current]
            for nb in adj.get(current, []):
                nb_point = _clamp_point_to_box(current_point, nb)
                new_cost = b_cost[current] + _euclid(current_point, nb_point)
                if nb not in b_cost or new_cost < b_cost[nb]:
                    b_cost[nb] = new_cost
                    b_came_from[nb] = current
                    b_detail[nb] = nb_point
                    priority = new_cost + _euclid(nb_point, source_point)
                    heapq.heappush(b_frontier, (priority, new_cost, nb))
            return

    while f_frontier and b_frontier:
        if best_meet is not None:
            if f_frontier[0][0] >= best_cost and b_frontier[0][0] >= best_cost:
                break

        if f_frontier[0][0] <= b_frontier[0][0]:
            _expand_forward()
        else:
            _expand_backward()

        if best_meet is not None and best_meet in f_came_from and best_meet in b_came_from:
            pass

    if best_meet is None:
        print("No path found between source and destination.")
        return [], explored

    box_path = _reconstruct_box_path(best_meet)
    for b in box_path:
        _record(b)
    path = _reconstruct_point_path(box_path)
    return path, explored
