"""
test_simulator.py — Phase 1 Tests for GridWorld + GridVisualizer

Run with:
    cd embodied-ai-robot
    python -m pytest tests/test_simulator.py -v

Or simply:
    python tests/test_simulator.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.simulator import GridWorld, Direction, WorldObject


# ── Helpers ────────────────────────────────────────────────────────────────────

def fresh_world() -> GridWorld:
    w = GridWorld(width=10, height=10)
    w.reset()
    return w


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_initial_robot_position():
    w = fresh_world()
    assert w.robot_x == 1 and w.robot_y == 1, "Robot should start at (1,1)"
    print("✓  test_initial_robot_position")


def test_default_objects_present():
    w = fresh_world()
    for name in ["red_box", "bottle", "chair", "table"]:
        assert name in w.objects, f"Missing object: {name}"
    print("✓  test_default_objects_present")


def test_move_right():
    w = fresh_world()
    w.set_robot_position(1, 1)
    result = w.move_robot(Direction.RIGHT)
    assert result["success"], "Should be able to move right from (1,1)"
    assert w.robot_x == 2 and w.robot_y == 1
    print("✓  test_move_right")


def test_move_blocked_by_obstacle():
    w = GridWorld(5, 5)
    w.reset()
    w.obstacles.clear()   # clear defaults
    w.objects.clear()
    w.add_obstacle(3, 1)
    w.set_robot_position(2, 1)
    result = w.move_robot(Direction.RIGHT)
    assert not result["success"], "Should be blocked by obstacle"
    assert result["reason"] == "obstacle"
    assert w.robot_x == 2, "Robot should not have moved"
    print("✓  test_move_blocked_by_obstacle")


def test_move_blocked_by_wall():
    w = fresh_world()
    w.set_robot_position(0, 0)
    result = w.move_robot(Direction.UP)
    assert not result["success"]
    assert result["reason"] == "wall"
    print("✓  test_move_blocked_by_wall")


def test_move_blocked_by_left_wall():
    w = fresh_world()
    w.set_robot_position(0, 5)
    result = w.move_robot(Direction.LEFT)
    assert not result["success"]
    assert result["reason"] == "wall"
    print("✓  test_move_blocked_by_left_wall")


def test_robot_facing_updates():
    w = fresh_world()
    for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
        w.move_robot(direction)
        assert w.robot_facing == direction
    print("✓  test_robot_facing_updates")


def test_step_counter():
    w = fresh_world()
    w.set_robot_position(3, 3)
    w.move_robot(Direction.RIGHT)
    w.move_robot(Direction.DOWN)
    assert w.steps == 2
    print("✓  test_step_counter")


def test_action_history():
    w = fresh_world()
    w.set_robot_position(3, 3)
    w.move_robot(Direction.RIGHT)
    w.move_robot(Direction.DOWN)
    assert w.action_history == ["move_right", "move_down"]
    print("✓  test_action_history")


def test_get_state_structure():
    w = fresh_world()
    state = w.get_state()
    assert "robot" in state
    assert "objects" in state
    assert "obstacles" in state
    assert "grid_size" in state
    assert state["robot"]["x"] == w.robot_x
    assert state["robot"]["y"] == w.robot_y
    print("✓  test_get_state_structure")


def test_get_object_position():
    w = fresh_world()
    pos = w.get_object_position("red_box")
    assert pos == (7, 2), f"Expected (7,2) got {pos}"
    none_pos = w.get_object_position("nonexistent")
    assert none_pos is None
    print("✓  test_get_object_position")


def test_is_adjacent_to_object():
    w = GridWorld(10, 10)
    w.reset()
    w.objects.clear()
    w.obstacles.clear()
    w.add_object(WorldObject("red_box", x=5, y=5))
    w.set_robot_position(4, 5)   # directly left → adjacent
    assert w.is_adjacent_to_object("red_box")
    w.set_robot_position(1, 1)   # far away → not adjacent
    assert not w.is_adjacent_to_object("red_box")
    print("✓  test_is_adjacent_to_object")


def test_reset_clears_state():
    w = fresh_world()
    w.set_robot_position(7, 7)
    w.move_robot(Direction.DOWN)
    w.reset()
    assert w.steps == 0
    assert w.action_history == []
    assert w.robot_x == 1 and w.robot_y == 1
    print("✓  test_reset_clears_state")


def test_visualizer_renders_without_error():
    """Smoke test: just ensure render() runs and returns a valid array."""
    from modules.visualizer import GridVisualizer
    import numpy as np
    w = fresh_world()
    viz = GridVisualizer(cell_size=60)
    img = viz.render(w)
    assert isinstance(img, np.ndarray), "render() should return a numpy array"
    assert img.ndim == 3 and img.shape[2] == 3, "Expected (H, W, 3) RGB array"
    assert img.size > 0
    print("✓  test_visualizer_renders_without_error")


def test_visualizer_save(tmp_path):
    """Check that save() creates a file on disk."""
    from modules.visualizer import GridVisualizer
    w = fresh_world()
    viz = GridVisualizer(cell_size=60)
    out = str(tmp_path / "test_render.png")
    path = viz.save(w, path=out)
    assert os.path.exists(path), f"Expected file at {path}"
    assert os.path.getsize(path) > 1000, "PNG file seems too small"
    print("✓  test_visualizer_save")


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())

    tests = [
        test_initial_robot_position,
        test_default_objects_present,
        test_move_right,
        test_move_blocked_by_obstacle,
        test_move_blocked_by_wall,
        test_move_blocked_by_left_wall,
        test_robot_facing_updates,
        test_step_counter,
        test_action_history,
        test_get_state_structure,
        test_get_object_position,
        test_is_adjacent_to_object,
        test_reset_clears_state,
        test_visualizer_renders_without_error,
        lambda: test_visualizer_save(tmp),
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗  {t.__name__}: {e}")
            failed += 1

    print(f"\n{'─'*40}")
    print(f"  {passed} passed   {failed} failed")
    print(f"{'─'*40}")
