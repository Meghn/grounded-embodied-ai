"""
planner.py — Pathfinding for the Embodied AI Robot Project
Phase 2: Planner

Two algorithms are implemented behind a single interface:

  BFS  (Breadth-First Search) — default
       Guarantees the shortest path by step count on an unweighted grid.
       Explores layer by layer: all cells 1 step away, then 2 steps, etc.
       O(W × H) time and space. Correct and simple — the right choice for
       a 10×10 grid.

  A*   — optional upgrade
       Uses a heuristic (Manhattan distance) to explore toward the goal
       first, skipping most of the grid. Much faster on large maps.
       Same optimal path as BFS on unweighted grids.

Both algorithms treat:
  - Grid boundaries as walls
  - Obstacle cells as impassable
  - Object cells as passable (robot navigates TO the object cell)

Public interface:
  planner = Planner(world)
  result  = planner.plan(start=(1,1), goal=(7,2))
  result.actions   → [Direction.RIGHT, Direction.RIGHT, ...]
  result.path      → [(1,1), (2,1), ..., (7,2)]
  result.success   → True | False
  result.algorithm → "bfs" | "astar"
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from modules.simulator import Direction, GridWorld, DIRECTION_DELTA


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    """
    Returned by Planner.plan().

    Attributes:
        success    : False if no path exists (goal blocked / unreachable)
        actions    : Ordered list of Directions to execute
        path       : Ordered list of (x, y) grid cells from start to goal
        algorithm  : Which algorithm produced this plan
        nodes_explored : How many cells were examined (useful for comparing BFS vs A*)
        error      : Human-readable reason if success=False
    """
    success:        bool
    actions:        list[Direction]     = field(default_factory=list)
    path:           list[tuple[int,int]]= field(default_factory=list)
    algorithm:      str                 = "bfs"
    nodes_explored: int                 = 0
    error:          str                 = ""

    def __repr__(self) -> str:
        if not self.success:
            return f"PlanResult(success=False, error='{self.error}')"
        dirs = [d.value for d in self.actions]
        return (
            f"PlanResult(success=True, steps={len(self.actions)}, "
            f"algorithm={self.algorithm}, nodes_explored={self.nodes_explored}, "
            f"actions={dirs})"
        )


# ── Planner ───────────────────────────────────────────────────────────────────

class Planner:
    """
    Path planner for a GridWorld.

    Usage:
        planner = Planner(world)

        # Navigate to a named object
        result = planner.plan_to_object("red_box")

        # Navigate to an arbitrary cell
        result = planner.plan(start=(1, 1), goal=(7, 2))

        # Use A* instead of BFS
        result = planner.plan(start=(1, 1), goal=(7, 2), algorithm="astar")

        if result.success:
            for action in result.actions:
                world.move_robot(action)
    """

    def __init__(self, world: GridWorld):
        self.world = world

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan_to_object(
        self,
        object_name: str,
        algorithm: str = "bfs",
    ) -> PlanResult:
        """
        Plan a path from the robot's current position to a named object.

        Args:
            object_name : Key in world.objects  (e.g. "red_box")
            algorithm   : "bfs" (default) or "astar"

        Returns:
            PlanResult
        """
        goal_pos = self.world.get_object_position(object_name)
        if goal_pos is None:
            return PlanResult(
                success=False,
                error=f"Object '{object_name}' not found in world",
            )

        start = (self.world.robot_x, self.world.robot_y)
        return self.plan(start=start, goal=goal_pos, algorithm=algorithm)

    def plan(
        self,
        start: tuple[int, int],
        goal:  tuple[int, int],
        algorithm: str = "bfs",
    ) -> PlanResult:
        """
        Plan a path between two grid cells.

        Args:
            start     : (x, y) starting cell
            goal      : (x, y) destination cell
            algorithm : "bfs" or "astar"

        Returns:
            PlanResult
        """
        # ── Trivial case ──────────────────────────────────────────────────────
        if start == goal:
            return PlanResult(
                success=True,
                actions=[],
                path=[start],
                algorithm=algorithm,
                nodes_explored=0,
            )

        # ── Validate positions ────────────────────────────────────────────────
        for label, pos in [("start", start), ("goal", goal)]:
            x, y = pos
            if not self.world._in_bounds(x, y):
                return PlanResult(
                    success=False,
                    error=f"{label} position {pos} is out of bounds",
                )

        if goal in self.world.obstacles:
            return PlanResult(
                success=False,
                error=f"Goal {goal} is an obstacle — unreachable",
            )

        # ── Dispatch ──────────────────────────────────────────────────────────
        if algorithm == "astar":
            return self._astar(start, goal)
        else:
            return self._bfs(start, goal)

    # ── BFS ───────────────────────────────────────────────────────────────────

    def _bfs(
        self,
        start: tuple[int, int],
        goal:  tuple[int, int],
    ) -> PlanResult:
        """
        Breadth-First Search.

        Data structures:
          queue   : deque of (x, y) cells to explore next
          visited : set of cells already seen (prevents re-exploration)
          parent  : dict mapping each cell → (parent_cell, direction_taken)
                    Used to reconstruct the path after goal is found.
        """
        queue:   deque[tuple[int,int]]                   = deque([start])
        visited: set[tuple[int,int]]                     = {start}
        parent:  dict[tuple[int,int], tuple | None]      = {start: None}
        nodes_explored = 0

        while queue:
            current = queue.popleft()
            nodes_explored += 1

            if current == goal:
                path, actions = self._reconstruct(parent, start, goal)
                return PlanResult(
                    success=True,
                    actions=actions,
                    path=path,
                    algorithm="bfs",
                    nodes_explored=nodes_explored,
                )

            for direction, (dx, dy) in DIRECTION_DELTA.items():
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if (
                    self.world._in_bounds(nx, ny)
                    and neighbor not in visited
                    and neighbor not in self.world.obstacles
                ):
                    visited.add(neighbor)
                    parent[neighbor] = (current, direction)
                    queue.append(neighbor)

        return PlanResult(
            success=False,
            algorithm="bfs",
            nodes_explored=nodes_explored,
            error=f"No path found from {start} to {goal} — goal may be surrounded by obstacles",
        )

    # ── A* ────────────────────────────────────────────────────────────────────

    def _astar(
        self,
        start: tuple[int, int],
        goal:  tuple[int, int],
    ) -> PlanResult:
        """
        A* Search with Manhattan distance heuristic.

        Data structures:
          open_heap : min-heap of (f_score, counter, cell)
                      counter breaks ties deterministically
          g_score   : dict mapping cell → cost from start (step count)
          parent    : dict mapping cell → (parent_cell, direction_taken)

        f(n) = g(n) + h(n)
          g(n) = steps taken so far
          h(n) = Manhattan distance to goal  (admissible: never overestimates)
        """
        def heuristic(pos: tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        counter = 0   # tie-breaker for heap stability
        open_heap: list[tuple[int, int, tuple[int,int]]] = []
        heapq.heappush(open_heap, (heuristic(start), counter, start))

        g_score: dict[tuple[int,int], int]          = {start: 0}
        parent:  dict[tuple[int,int], tuple | None] = {start: None}
        closed:  set[tuple[int,int]]                = set()
        nodes_explored = 0

        while open_heap:
            f, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)
            nodes_explored += 1

            if current == goal:
                path, actions = self._reconstruct(parent, start, goal)
                return PlanResult(
                    success=True,
                    actions=actions,
                    path=path,
                    algorithm="astar",
                    nodes_explored=nodes_explored,
                )

            for direction, (dx, dy) in DIRECTION_DELTA.items():
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if (
                    not self.world._in_bounds(nx, ny)
                    or neighbor in self.world.obstacles
                    or neighbor in closed
                ):
                    continue

                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent[neighbor]  = (current, direction)
                    f_score = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        return PlanResult(
            success=False,
            algorithm="astar",
            nodes_explored=nodes_explored,
            error=f"No path found from {start} to {goal} — goal may be surrounded by obstacles",
        )

    # ── Path reconstruction ───────────────────────────────────────────────────

    @staticmethod
    def _reconstruct(
        parent: dict,
        start:  tuple[int, int],
        goal:   tuple[int, int],
    ) -> tuple[list[tuple[int,int]], list[Direction]]:
        """
        Walk the parent dict backwards from goal to start,
        then reverse to get start → goal order.
        """
        path    = []
        actions = []
        current = goal

        while current != start:
            prev_cell, direction = parent[current]
            path.append(current)
            actions.append(direction)
            current = prev_cell

        path.append(start)
        path.reverse()
        actions.reverse()
        return path, actions
