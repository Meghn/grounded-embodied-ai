"""
demo_phase2.py — Interactive Phase 2 Demo
Run this to see the Planner + Execution Engine in action.

Usage:
    cd embodied-ai-robot
    python demo_phase2.py

What it demonstrates:
  1. BFS planning to each object — shows path length and nodes explored
  2. A* planning — compares nodes explored vs BFS
  3. Full execution: robot navigates step-by-step to "red_box"
  4. Path overlay image saved to assets/grid_path.png
  5. Full test suite (21 tests)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.simulator        import GridWorld, Direction
from modules.planner          import Planner
from modules.visualizer       import GridVisualizer
from modules.execution_engine import ExecutionEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP  = "─" * 52
DSEP = "═" * 52


def section(title: str) -> None:
    print(f"\n{DSEP}")
    print(f"  {title}")
    print(DSEP)


def on_step(step_num: int, state: dict) -> None:
    r = state["robot"]
    print(
        f"    step {step_num:>2}  │  "
        f"robot @ ({r['x']}, {r['y']})  "
        f"facing {r['facing']:<5}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{DSEP}")
    print("  EMBODIED AI — Phase 2 Demo")
    print("  Planner (BFS + A*) + Execution Engine")
    print(DSEP)

    world = GridWorld(10, 10)
    world.reset()
    viz     = GridVisualizer(cell_size=64)
    planner = Planner(world)

    # ── 1. Plan to every object with BFS ──────────────────────────────────────
    section("1. BFS  —  planning to all objects")
    print(f"  {'Object':<12}  {'Steps':>5}  {'Nodes':>6}  {'Path'}")
    print(f"  {SEP}")
    for name in ["red_box", "bottle", "chair", "table"]:
        result = planner.plan_to_object(name, algorithm="bfs")
        if result.success:
            path_str = " → ".join(f"({x},{y})" for x, y in result.path[:4])
            if len(result.path) > 4:
                path_str += f" ... ({result.path[-1][0]},{result.path[-1][1]})"
            print(f"  {name:<12}  {len(result.actions):>5}  {result.nodes_explored:>6}  {path_str}")
        else:
            print(f"  {name:<12}  FAILED — {result.error}")

    # ── 2. BFS vs A* comparison ───────────────────────────────────────────────
    section("2. BFS vs A*  —  efficiency comparison")
    print(f"  {'Object':<12}  {'BFS steps':>9}  {'BFS nodes':>9}  {'A* steps':>8}  {'A* nodes':>8}")
    print(f"  {SEP}")
    for name in ["red_box", "bottle", "chair", "table"]:
        bfs   = planner.plan_to_object(name, algorithm="bfs")
        astar = planner.plan_to_object(name, algorithm="astar")
        print(
            f"  {name:<12}  {len(bfs.actions):>9}  {bfs.nodes_explored:>9}  "
            f"{len(astar.actions):>8}  {astar.nodes_explored:>8}"
        )
    print(f"\n  A* explores fewer nodes by skipping cells away from the goal.")

    # ── 3. Save path overlay image ────────────────────────────────────────────
    section("3. Path overlay  —  assets/grid_path.png")
    plan = planner.plan_to_object("red_box")
    engine_preview = ExecutionEngine(
        world=world, planner=planner, visualizer=viz,
        save_frames=False, log_path=None,
    )
    overlay_path = engine_preview.render_with_path(plan, save_path="assets/grid_path.png")
    print(f"  Saved → {overlay_path}")
    print(f"  Plan  : {len(plan.actions)} steps, {plan.nodes_explored} nodes explored")
    print(f"  Path  : {plan.path[:3]} ... {plan.path[-1]}")

    # ── 4. Execute: robot walks to red_box ────────────────────────────────────
    section("4. Execution  —  navigating to 'red_box'")
    world.reset()
    engine_red_box = ExecutionEngine(
        world=world,
        planner=Planner(world),
        visualizer=viz,
        step_delay=0,
        save_frames=True,
        frame_dir="assets/frames/red_box",
        log_path="logs/session_logs.jsonl",
        on_step=on_step,
    )

    print(f"  Start : ({world.robot_x}, {world.robot_y})")
    print(f"  Goal  : {world.get_object_position('red_box')}")
    print(f"  {SEP}")

    result = engine_red_box.execute_goal("red_box", algorithm="bfs")

    print(f"  {SEP}")
    print(f"  {result.summary()}")
    print(f"  Frames → assets/frames/red_box/  ({result.actions_taken + 1} files)")
    print(f"  Log    → logs/session_logs.jsonl")

    # ── 5. Execute: robot walks to bottle ─────────────────────────────────────
    section("5. Execution  —  navigating to 'bottle'")
    world.reset()
    engine_bottle = ExecutionEngine(
        world=world,
        planner=Planner(world),
        visualizer=viz,
        step_delay=0,
        save_frames=True,
        frame_dir="assets/frames/bottle",
        log_path="logs/session_logs.jsonl",
        on_step=on_step,
    )

    print(f"  Start : ({world.robot_x}, {world.robot_y})")
    print(f"  Goal  : {world.get_object_position('bottle')}")
    print(f"  {SEP}")

    result2 = engine_bottle.execute_goal("bottle", algorithm="astar")
    print(f"  {SEP}")
    print(f"  {result2.summary()}")
    print(f"  Frames → assets/frames/bottle/  ({result2.actions_taken + 1} files)")

    # ── 6. Save final world render ────────────────────────────────────────────
    section("6. Final render  —  assets/grid_render.png")
    world.reset()
    viz.save(world, path="assets/grid_render.png")
    print(f"  Saved clean world render → assets/grid_render.png")

    # ── 7. Run tests ──────────────────────────────────────────────────────────
    section("7. Test suite")
    os.system(f"{sys.executable} tests/test_planner.py")


if __name__ == "__main__":
    main()
