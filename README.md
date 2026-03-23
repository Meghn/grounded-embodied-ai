<div align="center">

<br/>

# 🤖 Grounded Embodied AI

### Language-Conditioned Perception, Planning, and Action in a Simulated 2D Environment

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-15%20passing-22c55e?style=flat-square)](#testing)

<br/>

> *A complete sense → reason → act loop grounding natural language commands in a perception-driven, navigable 2D world — implemented from first principles, without ROS, Gazebo, or RL.*

<br/>

</div>

---

## Overview

**Grounded Embodied AI** is an end-to-end system where a robot agent interprets free-form natural language commands, perceives its environment through computer vision, plans a collision-free path, and executes actions in a simulated 2D grid world — all wired together through a production-grade API and interactive UI.

The project is architecturally faithful to real embodied AI research pipelines: the separation between perception, world modeling, language grounding, and motor execution mirrors systems like SayCan (Ahn et al., 2022), RT-2 (Brohan et al., 2023), and FILM (Min et al., 2022) — scaled to a form you can run locally in under a minute.

```
User: "Go to the red box."
        │
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  LLM Interpreter│────▶│ Perception Module │────▶│    A* Planner   │
│  GPT-4o → JSON  │     │ OpenCV / YOLOv8  │     │  BFS (v1) / A*  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                         ┌─────────────────┐              ▼
                         │  Visualizer     │◀────  Execution Engine
                         │  Matplotlib     │       Step-by-step loop
                         └─────────────────┘
```

**What makes this non-trivial:**
- The robot does not cheat. It reads the environment through a rendered image, not through internal simulator state. Perception outputs drive planning — exactly as in real robotics.
- The LLM is not a chatbot wrapper. It is a structured command interpreter with typed Pydantic outputs, failure handling, and a system prompt designed for grounded task decomposition.
- The system is observable. Every command, LLM response, perception detection, and action is logged in structured JSON — production telemetry on a research prototype.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [The Sense–Reason–Act Loop](#the-sensereasonact-loop)
- [Module Reference](#module-reference)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Running the Full Stack](#running-the-full-stack)
- [Configuration](#configuration)
- [Testing](#testing)
- [Perception Deep Dive](#perception-deep-dive)
- [LLM Grounding Design](#llm-grounding-design)
- [Roadmap](#roadmap)
- [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
- [Related Work](#related-work)
- [Citation](#citation)

---

## System Architecture

The system is organized into four tiers, each with a clean interface contract to the tier below it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1 — Interface Layer                                           │
│  Streamlit UI  ──────────────▶  FastAPI Backend (/command)          │
│  Natural language input         Request validation · routing        │
└─────────────────────────────────────┬───────────────────────────────┘
                                       │ POST {"command": str}
┌─────────────────────────────────────▼───────────────────────────────┐
│  TIER 2 — Reasoning Layer                                           │
│  LLM Interpreter (GPT-4o)                                           │
│  Prompt engineering · structured output · Pydantic validation       │
│  Output: {"goal": "red_box"}  ──or──  {"actions": [...]}            │
└─────────────────────────────────────┬───────────────────────────────┘
                          ┌───────────┴───────────┐
┌─────────────────────────▼──────┐   ┌────────────▼────────────────── ┐
│  TIER 3a — Perception          │   │  TIER 3b — Planning             │
│  GridVisualizer renders scene  │   │  BFS / A* on occupancy grid     │
│  OpenCV detects objects        │   │  Obstacle-aware path to goal    │
│  Returns {name, position}      │   │  Returns [Direction, ...]       │
└────────────────────────────────┘   └─────────────────────────────────┘
                          └───────────┬───────────┘
┌─────────────────────────────────────▼───────────────────────────────┐
│  TIER 4 — Execution + Simulation Layer                              │
│  Execution Engine  ──▶  GridWorld Simulator  ──▶  Visualizer        │
│  Steps actions          Robot + object state     PNG render         │
│  └──▶ Logger (JSONL) records every tick                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Properties

| Property | Decision | Rationale |
|---|---|---|
| Perception reads images | Visualizer renders → Perception reads image pixels | Faithful to real robot camera pipelines |
| Typed LLM outputs | Pydantic models, not raw JSON parsing | Fail-fast, observable, testable |
| Module isolation | Each module has a single public interface | Swap OpenCV for YOLO without changing the planner |
| Stateless API | `/command` endpoint is fully stateless | Enables horizontal scaling and replay |
| Structured logging | JSONL append-only logs | Production observability, replayable sessions |

---

## The Sense–Reason–Act Loop

This is the foundational concept in embodied AI. Every robotic system — from a Roomba to a humanoid — operates some variant of this loop.

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    ┌──────────┐   image   ┌───────────┐  detections     │
│    │Simulator │──────────▶│Perception │─────────────┐   │
│    └────▲─────┘           └───────────┘             │   │
│         │                                           ▼   │
│         │ execute        ┌───────────┐  goal    ┌──────┐│
│         └────────────────│ Execution │◀─────────│ LLM  ││
│                          │  Engine   │          └──────┘│
│                          └─────▲─────┘                  │
│                                │ actions                 │
│                          ┌─────┴─────┐                  │
│                          │  Planner  │                  │
│                          └───────────┘                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Why this matters:** The robot cannot introspect the simulator's Python state to find objects. It must observe the rendered image, detect objects visually, and use those coordinates for navigation. The indirection is intentional — it makes the perception module non-optional and architecturally honest.

---

## Module Reference

### `modules/simulator.py` — `GridWorld`

The authoritative world state. No AI logic; pure environment mechanics.

```python
from modules.simulator import GridWorld, Direction, WorldObject

world = GridWorld(width=10, height=10)
world.reset()

# Move the robot
result = world.move_robot(Direction.RIGHT)
# → {"success": True, "reason": "moved", "position": (2, 1), "steps": 1}

# Query state
world.get_state()
# → {"robot": {"x": 2, "y": 1, "facing": "right"}, "objects": {...}, ...}

world.get_object_position("red_box")
# → (7, 2)
```

**Default world layout (10×10):**

```
  0 1 2 3 4 5 6 7 8 9
0 . . . . . . . . . .
1 . R . . . . . . . .     R = Robot (1,1)   facing right
2 . . . . . ■ . ● . .     ● = red_box (7,2)
3 . . . ■ . ■ . . . .     ■ = obstacle
4 . . . . . ■ . . . .
5 . . . . . ■ . . . .
6 . . . . ◆ ■ . . . .     ◆ = bottle (4,6)
7 . . . . . . . . ▲ .     ▲ = table (8,7)
8 . . ▽ . . . ■ ■ . .     ▽ = chair (2,8)
9 . . . . . . . . . .
```

**Public interface:**

| Method | Signature | Description |
|---|---|---|
| `reset()` | `→ None` | Restore defaults, clear history |
| `move_robot()` | `(Direction) → dict` | Attempt movement, return result |
| `get_state()` | `→ dict` | Full world snapshot |
| `get_object_position()` | `(str) → tuple \| None` | Object coordinates |
| `is_adjacent_to_object()` | `(str) → bool` | Proximity check |
| `set_robot_position()` | `(int, int) → None` | Teleport (testing / reset) |

---

### `modules/visualizer.py` — `GridVisualizer`

Renders `GridWorld` state to a Matplotlib image. This image feeds directly into the Perception module — the visualizer is the robot's "camera".

```python
from modules.visualizer import GridVisualizer

viz = GridVisualizer(cell_size=64)
img_array = viz.render(world)   # → np.ndarray (H, W, 3), uint8
viz.save(world, "assets/grid_render.png")
```

**Rendering features:**
- Distinct color per object type (configurable per `WorldObject`)
- Robot rendered as a teal circle with a directional arrow indicating facing
- Obstacle cells marked with an `×` crosshatch pattern
- Step counter and position info in the figure title
- Legend row below the grid

---

### `modules/perception.py` — `PerceptionModule` *(Phase 3)*

Analyzes rendered scene images to detect objects and return their grid positions.

```python
from modules.perception import PerceptionModule

perception = PerceptionModule(backend="opencv")   # or "yolo"
detections = perception.detect("assets/grid_render.png")

# → [
#     {"object": "red_box",  "position": [7, 2], "confidence": 1.0},
#     {"object": "bottle",   "position": [4, 6], "confidence": 1.0},
#     {"object": "chair",    "position": [2, 8], "confidence": 1.0},
#     {"object": "table",    "position": [8, 7], "confidence": 1.0},
#   ]
```

**Backend v1 — OpenCV color segmentation:**
Each object has a registered HSV color range. `cv2.inRange()` isolates each range, `cv2.findContours()` extracts regions, and centroid pixel coordinates are mapped back to grid cells via the known cell size and padding.

**Backend v2 — YOLOv8 (planned):**
Swap `backend="yolo"` to use a fine-tuned YOLOv8n model. The public interface is identical — the planner never knows which backend is active.

---

### `modules/planner.py` — `Planner` *(Phase 2)*

Computes collision-free paths from robot position to goal.

```python
from modules.planner import Planner

planner = Planner(world)
actions = planner.navigate(start=(1, 1), goal=(7, 2))
# → [Direction.RIGHT, Direction.RIGHT, ..., Direction.UP]
```

**v1 — BFS:** Optimal for unweighted grids. Guarantees shortest path (by step count).

**v2 — A\*:** Heuristic-guided search. Required for larger maps (>20×20) where BFS becomes slow. Uses Manhattan distance as the admissible heuristic.

---

### `modules/llm_interpreter.py` — `LLMInterpreter` *(Phase 4)*

Converts natural language commands to structured, validated goal representations.

```python
from modules.llm_interpreter import LLMInterpreter

interpreter = LLMInterpreter()
result = interpreter.interpret("Go to the red box and then navigate to the chair")

# → GoalSequence(
#       goals=["red_box", "chair"],
#       raw_command="Go to the red box and then navigate to the chair"
#   )
```

**System prompt design principles:**
- Closed-vocabulary output (only valid object names accepted)
- Single-responsibility: command parsing only, no planning
- Structured failure mode: returns `{"error": "unrecognized_object"}` for unknown targets rather than hallucinating a goal

---

### `modules/execution_engine.py` — `ExecutionEngine` *(Phase 4)*

Drives the action loop, coordinating the planner output with simulator execution.

```python
from modules.execution_engine import ExecutionEngine

engine = ExecutionEngine(world, planner, visualizer, logger)
result = engine.execute_goal("red_box")
# Renders frame → logs step → moves robot → repeat
```

---

### `app/main.py` — FastAPI Backend *(Phase 5)*

```bash
POST /command
Content-Type: application/json

{"command": "Go to the red box"}

# 200 OK
{
  "status": "success",
  "goal": "red_box",
  "actions_taken": ["move_right", "move_right", "move_up", ...],
  "final_position": [7, 2],
  "steps": 8,
  "session_id": "f3a2b1c0"
}
```

---

## Project Structure

```
grounded-embodied-ai/
│
├── modules/
│   ├── __init__.py
│   ├── simulator.py          # GridWorld: state, movement, collision
│   ├── visualizer.py         # GridVisualizer: Matplotlib renderer
│   ├── perception.py         # PerceptionModule: OpenCV / YOLOv8 detection  [Phase 3]
│   ├── planner.py            # Planner: BFS / A* pathfinding                [Phase 2]
│   ├── llm_interpreter.py    # LLMInterpreter: command → structured goal    [Phase 4]
│   └── execution_engine.py   # ExecutionEngine: action loop coordinator     [Phase 4]
│
├── app/
│   ├── main.py               # FastAPI application                          [Phase 5]
│   ├── routers/
│   │   └── command.py        # POST /command route handler
│   └── schemas.py            # Pydantic request / response models
│
├── ui/
│   └── streamlit_app.py      # Streamlit frontend                           [Phase 5]
│
├── tests/
│   ├── test_simulator.py     # 15 unit tests for GridWorld + GridVisualizer ✅
│   ├── test_planner.py       # Pathfinding tests (Phase 2)
│   ├── test_perception.py    # Detection accuracy tests (Phase 3)
│   └── test_llm_interpreter.py  # LLM output validation tests (Phase 4)
│
├── logs/
│   └── session_logs.jsonl    # Append-only structured session telemetry
│
├── assets/
│   └── grid_render.png       # Latest environment render
│
├── demo_phase1.py            # Phase 1 interactive demo
├── config.py                 # Environment variables, grid config, model names
├── requirements.txt          # Phased dependency list
├── Dockerfile                # Container for deployment
├── .env.example              # Environment variable template
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- pip

### Phase 1 (available now)

```bash
# Clone
git clone https://github.com/your-username/grounded-embodied-ai.git
cd grounded-embodied-ai

# Install Phase 1 dependencies
pip install matplotlib numpy

# Run the demo — moves the robot, renders the world, runs all tests
python demo_phase1.py
```

Expected output:

```
══════════════════════════════════════════════════
  EMBODIED AI — Phase 1 Demo
  GridWorld Simulator + Visualizer
══════════════════════════════════════════════════

  Initial state
  Robot     : (1, 1)  facing right
  Steps     : 0
  Objects   : ['red_box', 'bottle', 'chair', 'table']

  Running movement sequence...
  move_right   [OK  ]  pos=(2,1)  reason=moved
  move_right   [OK  ]  pos=(3,1)  reason=moved
  ...
  move_right   [BLOCKED]  pos=(5,1)  reason=obstacle

  Render saved → assets/grid_render.png

  15 passed   0 failed
```

A rendered PNG of the grid world will be saved at `assets/grid_render.png`.

---

## Running the Full Stack

*(Phase 5 — coming soon)*

```bash
# Install all dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# → add your OPENAI_API_KEY

# Start the backend
uvicorn app.main:app --reload --port 8000

# In a separate terminal, start the UI
streamlit run ui/streamlit_app.py

# Open http://localhost:8501
# Type: "Go to the red box"
# Watch the robot navigate
```

---

## Configuration

```python
# config.py
GRID_WIDTH       = 10
GRID_HEIGHT      = 10
CELL_SIZE_PX     = 64

LLM_MODEL        = "gpt-4o"
LLM_TEMPERATURE  = 0.0          # Deterministic for command parsing

PERCEPTION_BACKEND = "opencv"   # "opencv" | "yolo"
YOLO_MODEL_PATH    = "models/yolov8n_gridworld.pt"

LOG_PATH         = "logs/session_logs.jsonl"
RENDER_PATH      = "assets/grid_render.png"
```

All sensitive values (API keys, deployment URLs) are loaded from `.env` and never committed.

---

## Testing

```bash
# Run Phase 1 test suite
python tests/test_simulator.py

# Run with pytest (optional)
pip install pytest
pytest tests/ -v
```

**Phase 1 test coverage:**

| Test | Status |
|---|---|
| `test_initial_robot_position` | ✅ |
| `test_default_objects_present` | ✅ |
| `test_move_right` | ✅ |
| `test_move_blocked_by_obstacle` | ✅ |
| `test_move_blocked_by_wall` | ✅ |
| `test_move_blocked_by_left_wall` | ✅ |
| `test_robot_facing_updates` | ✅ |
| `test_step_counter` | ✅ |
| `test_action_history` | ✅ |
| `test_get_state_structure` | ✅ |
| `test_get_object_position` | ✅ |
| `test_is_adjacent_to_object` | ✅ |
| `test_reset_clears_state` | ✅ |
| `test_visualizer_renders_without_error` | ✅ |
| `test_visualizer_save` | ✅ |

---

## Perception Deep Dive

The perception module is the most important technical component for understanding embodied AI. Here is why its design is non-obvious.

### Why perception reads images, not state

A naive implementation would have the planner query `world.get_object_position("red_box")` directly. This works, but it learns nothing. A real robot has no privileged access to a Python dictionary — it has a camera.

By forcing the perception module to read a rendered image:

1. The architecture is honest — the perception component does real work
2. You can swap the simulator for a real camera feed by changing one line
3. Perception errors propagate naturally (if the model misdetects an object, the robot navigates to the wrong place — exactly as in real systems)
4. You gain practical experience with the image → coordinate → action pipeline that is central to all modern robot learning

### OpenCV pipeline (v1)

```
Rendered grid image (640×640 px)
    │
    ▼
Convert BGR → HSV color space
    │
    ▼ for each registered object color range
cv2.inRange(image, lower_hsv, upper_hsv) → binary mask
    │
    ▼
cv2.findContours() → list of pixel blobs
    │
    ▼
Filter by minimum area (noise rejection)
    │
    ▼
cv2.moments() → centroid (cx, cy) in pixels
    │
    ▼
grid_x = (cx - padding) // cell_size
grid_y = (cy - padding) // cell_size
    │
    ▼
{"object": "red_box", "position": [grid_x, grid_y], "confidence": 1.0}
```

### YOLOv8 upgrade path (v2)

```python
# perception.py — swap backend, interface unchanged
class PerceptionModule:
    def detect(self, image_path: str) -> list[dict]:
        if self.backend == "opencv":
            return self._detect_opencv(image_path)
        elif self.backend == "yolo":
            return self._detect_yolo(image_path)   # same return type
```

The planner, execution engine, and LLM interpreter never change. This is the value of interface-driven design.

---

## LLM Grounding Design

Language grounding — mapping natural language to actionable robot commands — is an open research problem. This system implements a minimal but principled version.

### System prompt

```
You are a robot command parser for a 2D navigation task.

Valid objects in the environment: red_box, bottle, chair, table.

Given a user command, return ONLY a JSON object with key "goal" set to
the name of the target object.

If the command references multiple objects, return key "goals" as an
ordered list.

If no valid object is mentioned, return {"error": "unrecognized_object"}.

Do not explain. Do not add keys. Return only the JSON.
```

### Why this prompt works

- **Closed vocabulary:** The model cannot hallucinate an object name that doesn't exist in the world
- **Deterministic temperature:** `temperature=0.0` ensures identical outputs for identical commands — essential for reproducible testing
- **Single responsibility:** The LLM does command parsing only. Planning is separated. This prevents the LLM from outputting `{"actions": ["move_forward_3_steps"]}` — a leaky abstraction that embeds planning assumptions in the language layer
- **Explicit failure mode:** `{"error": ...}` responses are caught by Pydantic validation and returned to the user as actionable messages, not silent failures

### Comparison with related approaches

| System | Grounding approach |
|---|---|
| SayCan (Google, 2022) | LLM scores candidate skills by language likelihood |
| FILM (Min et al., 2022) | Language-conditioned semantic map + local policy |
| This system | LLM → closed-vocabulary goal extraction → symbolic planner |

This system's approach is closest to classical task-and-motion planning augmented with a language front-end — which is interpretable, debuggable, and appropriate for the scale of the environment.

---

## Roadmap

### Phase 1 — Grid Simulator + Visualizer ✅
- 10×10 GridWorld with robot, obstacles, and named objects
- Directional movement with collision detection
- Matplotlib renderer with per-object colors and robot direction arrow
- 15-test suite with full coverage of movement logic

### Phase 2 — Planner *(in progress)*
- BFS pathfinding from robot position to any named object
- Obstacle-aware path computation
- Execution Engine: step through action sequences with render-at-each-tick
- Path visualization overlay on grid render

### Phase 3 — Perception Module
- OpenCV HSV-range detection pipeline
- Grid-coordinate recovery from pixel centroids
- Per-object confidence scores
- Unit tests with synthetic test images
- YOLOv8 backend (fine-tuned on synthetic gridworld data)

### Phase 4 — LLM Interpreter + Full Pipeline
- GPT-4o command interpreter with Pydantic-validated outputs
- Multi-goal command support ("go to the box, then the chair")
- End-to-end pipeline: natural language → robot action
- Structured JSONL session logging

### Phase 5 — Production Stack
- FastAPI REST backend with `/command` endpoint
- Streamlit UI with live grid render updates
- Docker containerization
- Deployment to Render / Google Cloud Run
- Session replay from logs

### Version 2 (future)
- A* planner for larger maps
- YOLOv8 perception backend
- Persistent world model across commands
- ReAct-style LLM reasoning loop (agent asks "what do I see?" before acting)

### Version 3 (research direction)
- PyBullet 3D simulation
- Real camera-to-YOLO perception (no simulator rendering shortcut)
- Vector memory store (FAISS / Chroma) for cross-session object recall
- Language-conditioned exploration under partial observability

---

## Design Decisions and Tradeoffs

**Why a 2D grid and not PyBullet / Gazebo?**

Complexity budget. The goal of this phase is to build and understand the full sense → reason → act pipeline. A 3D simulator would spend 80% of development time on environment setup, physics tuning, and sensor configuration — and 20% on the AI architecture. The grid world inverts that ratio. The upgrade path to 3D is explicit in the roadmap and requires changing only the simulator and visualizer, not the LLM, planner, or perception interfaces.

**Why BFS before A*?**

BFS is optimal on unweighted grids and has no hyperparameters (no heuristic to choose, no weight to tune). It is the correct baseline. A* is an optimization for larger state spaces — adding it before it's needed is premature.

**Why OpenCV before YOLOv8?**

Learning sequence. OpenCV color segmentation makes the coordinate-recovery math explicit: you see exactly how pixel space maps to grid space. YOLO abstracts that away into a bounding box. Understanding the lower-level pipeline makes the higher-level model less of a black box.

**Why structured LLM outputs instead of free-form text parsing?**

Reliability. Free-form LLM output parsed with regex fails on paraphrases, punctuation variation, and capitalization. Pydantic-validated JSON fails loudly and predictably — which is far more useful in a debugging context than silently wrong behavior.

---

## Related Work

This project is designed to be conceptually aligned with the following lines of research. Reading these alongside the code will significantly deepen your understanding.

**Foundational:**
- Brooks, R. A. (1986). A robust layered control system for a mobile robot. *IEEE Journal on Robotics and Automation*.
- Arkin, R. C. (1998). *Behavior-Based Robotics*. MIT Press.

**Language-conditioned robot behavior:**
- Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *arXiv:2204.01691*. *(SayCan)*
- Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *arXiv:2307.15818*.
- Min, S., et al. (2022). FILM: Following Instructions in Language with Modular Methods. *ICLR 2022*.

**Perception for robotics:**
- Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR 2016*.
- Jocher, G., et al. (2023). Ultralytics YOLOv8. *(software)*

**Planning:**
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*.

---

## Citation

If you use this repository for coursework, research, or as a reference implementation, please cite:

```bibtex
@software{grounded_embodied_ai_2025,
  author    = {Your Name},
  title     = {Grounded Embodied AI: Language-Conditioned Perception, Planning, and Action},
  year      = {2025},
  url       = {https://github.com/your-username/grounded-embodied-ai},
  note      = {End-to-end embodied AI system: LLM grounding + computer vision
               perception + BFS/A* planning + 2D grid simulation}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for full text.

---

<div align="center">

Built systematically, phase by phase — because understanding every layer is the point.

</div>
