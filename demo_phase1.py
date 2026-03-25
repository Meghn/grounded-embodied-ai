"""
demo_phase1.py — Interactive Phase 1 Demo
Run this script to see the GridWorld + Visualizer in action.

Usage:
    cd embodied-ai-robot
    python demo_phase1.py

What it does:
  1. Creates a 10×10 GridWorld and resets it to defaults
  2. Prints the initial world state
  3. Moves the robot through a short manual sequence
  4. Saves a PNG render of the final state to assets/grid_render.png
  5. Prints all test results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.simulator import GridWorld, Direction
from modules.visualizer import GridVisualizer


def print_state(world: GridWorld, label: str = "") -> None:
    state = world.get_state()
    r = state["robot"]
    print(f"\n{'─' * 50}")
    if label:
        print(f"  {label}")
    print(f"  Robot     : ({r['x']}, {r['y']})  facing {r['facing']}")
    print(f"  Steps     : {state['steps']}")
    print(f"  Objects   : {list(state['objects'].keys())}")
    print(f"  Obstacles : {len(state['obstacles'])} cells")
    print(f"{'─' * 50}")


def move_and_report(world: GridWorld, direction: Direction) -> None:
    result = world.move_robot(direction)
    status = "OK  " if result["success"] else "BLOCKED"
    print(
        f"  move_{direction.value:<6}  [{status}]  "
        f"pos=({result['position'][0]},{result['position'][1]})  "
        f"reason={result['reason']}"
    )


def main():
    print("\n" + "═" * 50)
    print("  EMBODIED AI — Phase 1 Demo")
    print("  GridWorld Simulator + Visualizer")
    print("═" * 50)

    # ── 1. Create and reset world ──────────────────────────────────────────────
    world = GridWorld(width=10, height=10)
    world.reset()
    print_state(world, "Initial state")

    # ── 2. Manual movement sequence ───────────────────────────────────────────
    print("\n  Running movement sequence...\n")

    moves = [
        Direction.RIGHT,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.RIGHT,
        Direction.UP,       # this should work (clear cell)
        Direction.RIGHT,    # heading toward obstacle column
        Direction.RIGHT,    # hit obstacle at (5, y)?
        Direction.LEFT,
    ]

    for d in moves:
        move_and_report(world, d)

    print_state(world, "After movement sequence")

    # ── 3. Demonstrate wall blocking ──────────────────────────────────────────
    print("\n  Demonstrating wall-blocking (moving robot to corner)...")
    world.set_robot_position(0, 0)
    move_and_report(world, Direction.UP)    # should block (wall)
    move_and_report(world, Direction.LEFT)  # should block (wall)
    move_and_report(world, Direction.DOWN)  # should succeed
    move_and_report(world, Direction.RIGHT) # should succeed

    print_state(world, "After wall tests")

    # ── 4. Save render ────────────────────────────────────────────────────────
    world.reset()  # reset for a clean final image
    viz = GridVisualizer(cell_size=64)
    out_path = "assets/grid_render.png"
    saved = viz.save(world, path=out_path, show_title=True)
    print(f"\n  Render saved → {saved}")

    # ── 5. Run tests ──────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  Running test suite...")
    print("═" * 50 + "\n")
    os.system(f"{sys.executable} tests/test_simulator.py")


if __name__ == "__main__":
    main()
