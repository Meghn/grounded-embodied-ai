"""
simulator.py — 2D Grid World for the Embodied AI Robot Project
Phase 1: Environment Simulator

The GridWorld holds all state for the simulation:
  - A grid of cells (empty, obstacle, or object)
  - Robot position and facing direction
  - Named objects with positions
  - Movement logic with collision detection

Grid coordinate system:
  (0, 0) = top-left corner
  x increases rightward, y increases downward
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import copy


# ── Cell types ────────────────────────────────────────────────────────────────

class CellType(Enum):
    EMPTY    = "empty"
    OBSTACLE = "obstacle"
    ROBOT    = "robot"
    OBJECT   = "object"


# ── Direction ─────────────────────────────────────────────────────────────────

class Direction(Enum):
    UP    = "up"
    DOWN  = "down"
    LEFT  = "left"
    RIGHT = "right"

# Maps each direction to a (dx, dy) delta
DIRECTION_DELTA: dict[Direction, tuple[int, int]] = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}


# ── Object definition ─────────────────────────────────────────────────────────

@dataclass
class WorldObject:
    name: str           # e.g. "red_box"
    x: int
    y: int
    color: tuple[int, int, int] = (200, 50, 50)   # RGB for the visualizer
    symbol: str = "?"                              # Single character for text debug


# ── Grid World ────────────────────────────────────────────────────────────────

class GridWorld:
    """
    A 2D grid-based simulated environment.

    Usage:
        world = GridWorld(width=10, height=10)
        world.reset()
        world.move_robot(Direction.RIGHT)
        state = world.get_state()
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.width  = width
        self.height = height

        # Robot state
        self.robot_x: int = 0
        self.robot_y: int = 0
        self.robot_facing: Direction = Direction.RIGHT

        # Named objects in the world  {name: WorldObject}
        self.objects: dict[str, WorldObject] = {}

        # Obstacle set  {(x, y)}
        self.obstacles: set[tuple[int, int]] = set()

        # Action history for logging
        self.action_history: list[str] = []

        # Step counter
        self.steps: int = 0

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the world and place defaults."""
        self.objects.clear()
        self.obstacles.clear()
        self.action_history.clear()
        self.steps = 0
        self.robot_x = 1
        self.robot_y = 1
        self.robot_facing = Direction.RIGHT

        # Default objects
        self._place_default_objects()

        # Default obstacles (a small wall cluster)
        self._place_default_obstacles()

    def _place_default_objects(self) -> None:
        self.add_object(WorldObject("red_box",  x=7, y=2, color=(220, 60,  60),  symbol="R"))
        self.add_object(WorldObject("bottle",   x=4, y=6, color=(60,  120, 220), symbol="B"))
        self.add_object(WorldObject("chair",    x=2, y=8, color=(180, 140, 60),  symbol="C"))
        self.add_object(WorldObject("table",    x=8, y=7, color=(120, 80,  40),  symbol="T"))

    def _place_default_obstacles(self) -> None:
        wall = [(5, 3), (5, 4), (5, 5), (5, 6), (3, 3), (6, 8), (7, 8)]
        for pos in wall:
            self.obstacles.add(pos)

    def add_object(self, obj: WorldObject) -> None:
        self.objects[obj.name] = obj

    def add_obstacle(self, x: int, y: int) -> None:
        self.obstacles.add((x, y))

    def set_robot_position(self, x: int, y: int) -> None:
        if self._in_bounds(x, y):
            self.robot_x = x
            self.robot_y = y

    # ── Movement ───────────────────────────────────────────────────────────────

    def move_robot(self, direction: Direction) -> dict:
        """
        Move the robot one step in the given direction.

        Returns a result dict:
            {
              "success": bool,
              "reason":  str,          # "moved" | "wall" | "obstacle" | "object"
              "position": (x, y),
              "direction": str,
            }
        """
        dx, dy = DIRECTION_DELTA[direction]
        new_x = self.robot_x + dx
        new_y = self.robot_y + dy

        self.robot_facing = direction
        self.steps += 1
        self.action_history.append(f"move_{direction.value}")

        # Out-of-bounds check
        if not self._in_bounds(new_x, new_y):
            return self._move_result(False, "wall")

        # Obstacle check
        if (new_x, new_y) in self.obstacles:
            return self._move_result(False, "obstacle")

        # Object collision check (objects are passable in v1 — robot walks to them)
        obj_at = self._object_at(new_x, new_y)
        if obj_at:
            # Move onto object cell — treated as "arrived"
            self.robot_x = new_x
            self.robot_y = new_y
            return self._move_result(True, f"arrived:{obj_at}", adjacent=True)

        # Free move
        self.robot_x = new_x
        self.robot_y = new_y
        return self._move_result(True, "moved")

    def _move_result(
        self,
        success: bool,
        reason: str,
        adjacent: bool = False
    ) -> dict:
        return {
            "success":   success,
            "reason":    reason,
            "position":  (self.robot_x, self.robot_y),
            "direction": self.robot_facing.value,
            "steps":     self.steps,
        }

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a full snapshot of the world state."""
        return {
            "robot": {
                "x":       self.robot_x,
                "y":       self.robot_y,
                "facing":  self.robot_facing.value,
            },
            "objects": {
                name: {"x": obj.x, "y": obj.y}
                for name, obj in self.objects.items()
            },
            "obstacles": list(self.obstacles),
            "grid_size": {"width": self.width, "height": self.height},
            "steps":     self.steps,
        }

    def get_object_position(self, name: str) -> Optional[tuple[int, int]]:
        obj = self.objects.get(name)
        return (obj.x, obj.y) if obj else None

    def is_robot_at(self, x: int, y: int) -> bool:
        return self.robot_x == x and self.robot_y == y

    def is_adjacent_to_object(self, name: str) -> bool:
        obj = self.objects.get(name)
        if not obj:
            return False
        return abs(self.robot_x - obj.x) + abs(self.robot_y - obj.y) <= 1

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _object_at(self, x: int, y: int) -> Optional[str]:
        for name, obj in self.objects.items():
            if obj.x == x and obj.y == y:
                return name
        return None

    def __repr__(self) -> str:
        return (
            f"GridWorld({self.width}x{self.height}) "
            f"robot=({self.robot_x},{self.robot_y}) "
            f"objects={list(self.objects.keys())} "
            f"obstacles={len(self.obstacles)}"
        )
