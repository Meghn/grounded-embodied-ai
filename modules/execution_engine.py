"""
execution_engine.py — Action Loop for the Embodied AI Robot Project
Phase 2: Execution Engine

The ExecutionEngine bridges the Planner and the Simulator.
It takes a PlanResult (a sequence of Directions) and executes them
one step at a time, rendering and optionally saving a frame after each move.

Responsibilities:
  - Step through planned actions
  - Call world.move_robot() for each action
  - Render and save the grid after each step
  - Detect and handle unexpected blocks (obstacle appeared mid-execution)
  - Log each step with timing and outcome
  - Return a structured ExecutionResult

This is the "motor cortex" of the agent — it knows nothing about
language or vision, only about driving the robot through the world.
"""

from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from modules.simulator import GridWorld, Direction
from modules.planner   import Planner, PlanResult
from modules.visualizer import GridVisualizer


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """
    Returned by ExecutionEngine.execute().

    Attributes:
        success         : True if robot reached the goal
        goal_name       : Name of the target object (or "custom")
        goal_position   : (x, y) of intended goal
        final_position  : (x, y) of robot when execution ended
        actions_planned : Total actions in the plan
        actions_taken   : Actions successfully executed
        steps_log       : Per-step records for logging / replay
        elapsed_seconds : Wall-clock execution time
        error           : Reason for failure if success=False
    """
    success:          bool
    goal_name:        str                  = ""
    goal_position:    tuple[int, int]      = (0, 0)
    final_position:   tuple[int, int]      = (0, 0)
    actions_planned:  int                  = 0
    actions_taken:    int                  = 0
    steps_log:        list[dict]           = field(default_factory=list)
    elapsed_seconds:  float                = 0.0
    error:            str                  = ""

    def summary(self) -> str:
        if self.success:
            return (
                f"✓  Reached '{self.goal_name}' at {self.goal_position} "
                f"in {self.actions_taken} steps  ({self.elapsed_seconds:.2f}s)"
            )
        return (
            f"✗  Failed to reach '{self.goal_name}': {self.error} "
            f"({self.actions_taken}/{self.actions_planned} steps completed)"
        )


# ── Execution Engine ──────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Executes a planned sequence of actions in a GridWorld.

    Usage — simplest form:
        engine = ExecutionEngine(world, planner, visualizer)
        result = engine.execute_goal("red_box")
        print(result.summary())

    Usage — custom plan:
        plan   = planner.plan(start=(1,1), goal=(7,2))
        result = engine.execute(plan, goal_name="custom", goal_position=(7,2))

    Args:
        world       : GridWorld instance (will be mutated)
        planner     : Planner instance
        visualizer  : GridVisualizer instance
        step_delay  : Seconds to pause between steps (0 = no pause)
        save_frames : If True, save a PNG after every step
        frame_dir   : Directory for per-step PNG frames
        log_path    : Path to JSONL log file (None = no logging)
        on_step     : Optional callback(step_num, world_state) called after each step
    """

    def __init__(
        self,
        world:       GridWorld,
        planner:     Planner,
        visualizer:  GridVisualizer,
        step_delay:  float  = 0.0,
        save_frames: bool   = True,
        frame_dir:   str    = "assets/frames",
        log_path:    Optional[str] = "logs/session_logs.jsonl",
        on_step:     Optional[Callable] = None,
    ):
        self.world       = world
        self.planner     = planner
        self.visualizer  = visualizer
        self.step_delay  = step_delay
        self.save_frames = save_frames
        self.frame_dir   = frame_dir
        self.log_path    = log_path
        self.on_step     = on_step

        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if save_frames:
            Path(frame_dir).mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute_goal(
        self,
        object_name: str,
        algorithm:   str = "bfs",
    ) -> ExecutionResult:
        """
        High-level entry point: plan + execute to a named object.

        Args:
            object_name : e.g. "red_box"
            algorithm   : "bfs" (default) or "astar"

        Returns:
            ExecutionResult
        """
        goal_pos = self.world.get_object_position(object_name)
        if goal_pos is None:
            return ExecutionResult(
                success=False,
                goal_name=object_name,
                error=f"Object '{object_name}' does not exist in the world",
            )

        plan = self.planner.plan_to_object(object_name, algorithm=algorithm)

        if not plan.success:
            return ExecutionResult(
                success=False,
                goal_name=object_name,
                goal_position=goal_pos,
                error=plan.error,
            )

        return self.execute(plan, goal_name=object_name, goal_position=goal_pos)

    def execute(
        self,
        plan:          PlanResult,
        goal_name:     str              = "custom",
        goal_position: tuple[int, int]  = (0, 0),
    ) -> ExecutionResult:
        """
        Execute a pre-computed PlanResult step by step.

        Args:
            plan          : PlanResult from Planner.plan()
            goal_name     : Label for logging
            goal_position : Target (x, y) for arrival check

        Returns:
            ExecutionResult
        """
        if not plan.success or not plan.actions:
            # Already at goal
            return ExecutionResult(
                success=True,
                goal_name=goal_name,
                goal_position=goal_position,
                final_position=(self.world.robot_x, self.world.robot_y),
                actions_planned=0,
                actions_taken=0,
            )

        start_time = time.time()
        steps_log: list[dict] = []

        # ── Save initial frame ─────────────────────────────────────────────────
        self._save_frame(step=0, path=self._frame_path(0))

        # ── Step through actions ───────────────────────────────────────────────
        for i, direction in enumerate(plan.actions, start=1):
            move_result = self.world.move_robot(direction)

            step_record = {
                "step":      i,
                "action":    direction.value,
                "success":   move_result["success"],
                "reason":    move_result["reason"],
                "position":  list(move_result["position"]),
                "timestamp": time.time(),
            }
            steps_log.append(step_record)

            # Save frame
            frame_path = self._frame_path(i)
            self._save_frame(step=i, path=frame_path)

            # Fire callback if registered (used by demo to print progress)
            if self.on_step:
                self.on_step(i, self.world.get_state())

            # Unexpected block — plan is now invalid
            if not move_result["success"]:
                elapsed = time.time() - start_time
                result = ExecutionResult(
                    success=False,
                    goal_name=goal_name,
                    goal_position=goal_position,
                    final_position=(self.world.robot_x, self.world.robot_y),
                    actions_planned=len(plan.actions),
                    actions_taken=i - 1,
                    steps_log=steps_log,
                    elapsed_seconds=elapsed,
                    error=f"Blocked at step {i}: {move_result['reason']} at {move_result['position']}",
                )
                self._log(goal_name, plan, result)
                return result

            # Optional pacing delay
            if self.step_delay > 0:
                time.sleep(self.step_delay)

        # ── Execution complete ─────────────────────────────────────────────────
        elapsed = time.time() - start_time
        final_pos = (self.world.robot_x, self.world.robot_y)
        arrived   = (final_pos == goal_position)

        result = ExecutionResult(
            success=arrived,
            goal_name=goal_name,
            goal_position=goal_position,
            final_position=final_pos,
            actions_planned=len(plan.actions),
            actions_taken=len(plan.actions),
            steps_log=steps_log,
            elapsed_seconds=elapsed,
            error="" if arrived else f"Ended at {final_pos}, expected {goal_position}",
        )

        self._log(goal_name, plan, result)
        return result

    # ── Path overlay render ────────────────────────────────────────────────────

    def render_with_path(
        self,
        plan:      PlanResult,
        save_path: str = "assets/grid_path.png",
    ) -> str:
        """
        Save a render of the current world with the planned path overlaid
        as a dotted line — useful for inspecting the plan before execution.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Use the visualizer to get the base figure
        fig, ax = self.visualizer._make_figure(self.world)
        self.visualizer._draw(ax, self.world, show_title=False)

        # Overlay path dots
        if plan.path:
            xs = [p[0] for p in plan.path]
            ys = [p[1] for p in plan.path]

            # Draw path line
            ax.plot(
                xs, ys,
                color="#F59E0B",   # amber
                linewidth=1.8,
                linestyle="--",
                zorder=8,
                alpha=0.85,
            )

            # Draw step dots
            for step_num, (px, py) in enumerate(plan.path):
                ax.plot(px, py, "o",
                    color="#F59E0B",
                    markersize=4,
                    zorder=9,
                    alpha=0.9,
                )
                if step_num > 0:   # skip label on start cell
                    ax.text(
                        px + 0.18, py - 0.18,
                        str(step_num),
                        fontsize=5,
                        color="#92400E",
                        zorder=10,
                    )

            # Highlight start and goal
            sx, sy = plan.path[0]
            gx, gy = plan.path[-1]
            ax.plot(sx, sy, "o", color="#10B981", markersize=7, zorder=11)
            ax.plot(gx, gy, "*", color="#EF4444", markersize=10, zorder=11)

        ax.set_title(
            f"Planned path  |  {len(plan.actions)} steps  |  "
            f"algorithm: {plan.algorithm}  |  nodes explored: {plan.nodes_explored}",
            fontsize=8, color="#444", pad=6,
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight",
                    facecolor=(0.97, 0.97, 0.95))
        plt.close(fig)
        return os.path.abspath(save_path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _save_frame(self, step: int, path: str) -> None:
        if self.save_frames:
            self.visualizer.save(self.world, path=path, show_title=True)
        else:
            # Always save the latest render to the canonical path
            self.visualizer.save(self.world, path="assets/grid_render.png")

    def _frame_path(self, step: int) -> str:
        return os.path.join(self.frame_dir, f"step_{step:04d}.png")

    def _log(
        self,
        goal_name: str,
        plan:      PlanResult,
        result:    ExecutionResult,
    ) -> None:
        if not self.log_path:
            return
        record = {
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "goal":            goal_name,
            "goal_position":   list(result.goal_position),
            "algorithm":       plan.algorithm,
            "actions_planned": result.actions_planned,
            "actions_taken":   result.actions_taken,
            "success":         result.success,
            "final_position":  list(result.final_position),
            "elapsed_seconds": round(result.elapsed_seconds, 4),
            "error":           result.error,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
