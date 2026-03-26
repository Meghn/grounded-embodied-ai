"""
test_planner.py — Phase 2 Tests for Planner + ExecutionEngine

Run with:
    cd embodied-ai-robot
    python tests/test_planner.py

Or with pytest:
    pytest tests/test_planner.py -v
"""

import sys
import os
import tempfile
import pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.simulator      import GridWorld, Direction, WorldObject
from modules.planner        import Planner, PlanResult
from modules.visualizer     import GridVisualizer
from modules.execution_engine import ExecutionEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def open_world(width=10, height=10) -> GridWorld:
    """A completely empty world — no obstacles, no objects."""
    w = GridWorld(width, height)
    w.reset()
    w.obstacles.clear()
    w.objects.clear()
    w.set_robot_position(1, 1)
    return w


def default_world() -> GridWorld:
    """Full default world with obstacles and all four objects."""
    w = GridWorld(10, 10)
    w.reset()
    return w


def make_engine(world: GridWorld, tmp: pathlib.Path) -> ExecutionEngine:
    viz = GridVisualizer(cell_size=60)
    planner = Planner(world)
    return ExecutionEngine(
        world=world,
        planner=planner,
        visualizer=viz,
        step_delay=0,
        save_frames=True,
        frame_dir=str(tmp / "frames"),
        log_path=str(tmp / "test_log.jsonl"),
    )


# ══ PLANNER TESTS ═════════════════════════════════════════════════════════════

def test_bfs_open_straight_line():
    """BFS on a clear grid — straight right."""
    w = open_world()
    p = Planner(w)
    result = p.plan(start=(0, 0), goal=(4, 0), algorithm="bfs")
    assert result.success, result.error
    assert len(result.actions) == 4
    assert all(a == Direction.RIGHT for a in result.actions)
    print("✓  test_bfs_open_straight_line")


def test_bfs_open_diagonal_path():
    """BFS on a clear grid — diagonal (L-shaped shortest path)."""
    w = open_world()
    p = Planner(w)
    result = p.plan(start=(0, 0), goal=(3, 3), algorithm="bfs")
    assert result.success
    assert len(result.actions) == 6   # Manhattan distance = 3+3
    print("✓  test_bfs_open_diagonal_path")


def test_bfs_detours_around_obstacle():
    """BFS finds a path around a wall column."""
    w = open_world(10, 10)
    # Wall: column x=5 from y=0..7 — forces robot to go below
    for y in range(8):
        w.add_obstacle(5, y)
    p = Planner(w)
    result = p.plan(start=(0, 0), goal=(7, 0), algorithm="bfs")
    assert result.success, result.error
    # Path must exist; it can't go straight through x=5
    path_xs = [pos[0] for pos in result.path]
    assert 5 not in path_xs or any(pos[1] >= 8 for pos in result.path), \
        "Path should go around or below the wall"
    print("✓  test_bfs_detours_around_obstacle")


def test_bfs_no_path_fully_enclosed():
    """BFS returns failure when goal is completely enclosed."""
    w = open_world(6, 6)
    # Box the goal in
    goal = (3, 3)
    for ox, oy in [(2,3),(4,3),(3,2),(3,4)]:
        w.add_obstacle(ox, oy)
    p = Planner(w)
    result = p.plan(start=(0, 0), goal=goal)
    assert not result.success
    assert "No path" in result.error or "obstacle" in result.error
    print("✓  test_bfs_no_path_fully_enclosed")


def test_bfs_same_start_and_goal():
    """Already at goal — zero actions."""
    w = open_world()
    p = Planner(w)
    result = p.plan(start=(3, 3), goal=(3, 3))
    assert result.success
    assert result.actions == []
    assert result.path == [(3, 3)]
    print("✓  test_bfs_same_start_and_goal")


def test_bfs_goal_is_obstacle():
    """Goal on an obstacle cell → immediate failure."""
    w = open_world()
    w.add_obstacle(5, 5)
    p = Planner(w)
    result = p.plan(start=(0, 0), goal=(5, 5))
    assert not result.success
    assert "obstacle" in result.error.lower()
    print("✓  test_bfs_goal_is_obstacle")


def test_bfs_out_of_bounds_start():
    w = open_world()
    p = Planner(w)
    result = p.plan(start=(-1, 0), goal=(5, 5))
    assert not result.success
    print("✓  test_bfs_out_of_bounds_start")


def test_bfs_path_length_is_optimal():
    """BFS always produces the shortest path."""
    w = open_world()
    p = Planner(w)
    result = p.plan(start=(0, 0), goal=(9, 9))
    assert result.success
    assert len(result.actions) == 18   # 9 right + 9 down = 18
    print("✓  test_bfs_path_length_is_optimal")


def test_astar_matches_bfs_path_length():
    """A* produces same-length path as BFS on open grid."""
    w = open_world()
    p = Planner(w)
    bfs   = p.plan(start=(0, 0), goal=(9, 9), algorithm="bfs")
    astar = p.plan(start=(0, 0), goal=(9, 9), algorithm="astar")
    assert bfs.success and astar.success
    assert len(bfs.actions) == len(astar.actions), \
        f"BFS={len(bfs.actions)}, A*={len(astar.actions)}"
    print("✓  test_astar_matches_bfs_path_length")


def test_astar_explores_fewer_nodes_than_bfs():
    """A* should explore fewer nodes than BFS on a large open grid."""
    w = open_world(20, 20)
    p = Planner(w)
    bfs   = p.plan(start=(0, 0), goal=(19, 19), algorithm="bfs")
    astar = p.plan(start=(0, 0), goal=(19, 19), algorithm="astar")
    assert astar.nodes_explored <= bfs.nodes_explored, \
        f"A*={astar.nodes_explored} should be ≤ BFS={bfs.nodes_explored}"
    print("✓  test_astar_explores_fewer_nodes_than_bfs")


def test_plan_to_object_red_box():
    """plan_to_object resolves name → position correctly."""
    w = default_world()
    p = Planner(w)
    result = p.plan_to_object("red_box")
    assert result.success, result.error
    goal = w.get_object_position("red_box")
    assert result.path[-1] == goal
    print("✓  test_plan_to_object_red_box")


def test_plan_to_object_unknown():
    """plan_to_object returns failure for a nonexistent object."""
    w = default_world()
    p = Planner(w)
    result = p.plan_to_object("invisible_item")
    assert not result.success
    assert "not found" in result.error
    print("✓  test_plan_to_object_unknown")


def test_plan_to_all_objects():
    """BFS can find paths to every default object from start."""
    w = default_world()
    p = Planner(w)
    for name in ["red_box", "bottle", "chair", "table"]:
        result = p.plan_to_object(name)
        assert result.success, f"Failed to find path to {name}: {result.error}"
    print("✓  test_plan_to_all_objects")


def test_path_contains_no_obstacles():
    """Every cell in every BFS path should be obstacle-free."""
    w = default_world()
    p = Planner(w)
    for name in ["red_box", "bottle", "chair", "table"]:
        result = p.plan_to_object(name)
        for cell in result.path[:-1]:   # exclude goal (is the object cell)
            assert cell not in w.obstacles, \
                f"Path for {name} passes through obstacle at {cell}"
    print("✓  test_path_contains_no_obstacles")


# ══ EXECUTION ENGINE TESTS ════════════════════════════════════════════════════

def test_execute_goal_reaches_target(tmp_path):
    w = default_world()
    engine = make_engine(w, tmp_path)
    result = engine.execute_goal("red_box")
    assert result.success, result.error
    goal = default_world().get_object_position("red_box")
    assert result.final_position == goal
    print("✓  test_execute_goal_reaches_target")


def test_execute_goal_unknown_object(tmp_path):
    w = default_world()
    engine = make_engine(w, tmp_path)
    result = engine.execute_goal("phantom_object")
    assert not result.success
    print("✓  test_execute_goal_unknown_object")


def test_execute_logs_all_steps(tmp_path):
    w = default_world()
    engine = make_engine(w, tmp_path)
    result = engine.execute_goal("bottle")
    assert result.success
    assert len(result.steps_log) == result.actions_taken
    for rec in result.steps_log:
        assert "step" in rec and "action" in rec and "position" in rec
    print("✓  test_execute_logs_all_steps")


def test_execute_saves_frames(tmp_path):
    w = default_world()
    engine = make_engine(w, tmp_path)
    result = engine.execute_goal("chair")
    assert result.success
    frames = list((tmp_path / "frames").glob("step_*.png"))
    # Should have initial frame + one per step
    assert len(frames) == result.actions_taken + 1, \
        f"Expected {result.actions_taken + 1} frames, got {len(frames)}"
    print("✓  test_execute_saves_frames")


def test_execute_writes_jsonl_log(tmp_path):
    w = default_world()
    engine = make_engine(w, tmp_path)
    engine.execute_goal("table")
    log_file = tmp_path / "test_log.jsonl"
    assert log_file.exists()
    import json
    with open(log_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) >= 1
    rec = records[0]
    assert "goal" in rec and "success" in rec and "actions_taken" in rec
    print("✓  test_execute_writes_jsonl_log")


def test_execute_all_objects_sequentially(tmp_path):
    """Execute navigation to all four objects in sequence."""
    w = default_world()
    engine = make_engine(w, tmp_path)
    for name in ["red_box", "bottle", "chair", "table"]:
        w.reset()   # reset between goals so robot starts fresh each time
        engine.world = w
        engine.planner.world = w
        result = engine.execute_goal(name)
        assert result.success, f"Failed to reach {name}: {result.error}"
    print("✓  test_execute_all_objects_sequentially")


def test_render_with_path(tmp_path):
    w = default_world()
    viz = GridVisualizer(cell_size=60)
    p = Planner(w)
    engine = ExecutionEngine(w, p, viz,
                             save_frames=True,
                             frame_dir=str(tmp_path / "frames"),
                             log_path=str(tmp_path / "log.jsonl"))
    plan = p.plan_to_object("red_box")
    out_path = str(tmp_path / "path_overlay.png")
    saved = engine.render_with_path(plan, save_path=out_path)
    assert os.path.exists(saved)
    assert os.path.getsize(saved) > 1000
    print("✓  test_render_with_path")


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tmp = pathlib.Path(tempfile.mkdtemp())

    tests = [
        # Planner
        test_bfs_open_straight_line,
        test_bfs_open_diagonal_path,
        test_bfs_detours_around_obstacle,
        test_bfs_no_path_fully_enclosed,
        test_bfs_same_start_and_goal,
        test_bfs_goal_is_obstacle,
        test_bfs_out_of_bounds_start,
        test_bfs_path_length_is_optimal,
        test_astar_matches_bfs_path_length,
        test_astar_explores_fewer_nodes_than_bfs,
        test_plan_to_object_red_box,
        test_plan_to_object_unknown,
        test_plan_to_all_objects,
        test_path_contains_no_obstacles,
        # Execution engine
        lambda: test_execute_goal_reaches_target(tmp),
        lambda: test_execute_goal_unknown_object(tmp),
        lambda: test_execute_logs_all_steps(tmp),
        lambda: test_execute_saves_frames(tmp),
        lambda: test_execute_writes_jsonl_log(tmp),
        lambda: test_execute_all_objects_sequentially(tmp),
        lambda: test_render_with_path(tmp),
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            name = getattr(t, "__name__", repr(t))
            print(f"✗  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'─'*42}")
    print(f"  {passed} passed   {failed} failed")
    print(f"{'─'*42}")
