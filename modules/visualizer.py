"""
visualizer.py — Grid World Renderer for the Embodied AI Robot Project
Phase 1: Visualization Module

Renders a GridWorld state as a clean Matplotlib image.
The rendered image is both:
  - Saved to disk (assets/grid_render.png) for the Streamlit UI
  - Returned as a numpy array for programmatic use

Color legend:
  White      → empty cell
  Dark gray  → obstacle / wall
  Cyan/teal  → robot (with a directional arrow)
  Per-object → each WorldObject has its own RGB color

This module has no knowledge of the AI pipeline — it only reads
GridWorld state and draws it. Keep it that way.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for servers & Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from pathlib import Path
from typing import Optional

# We import only the type; the actual instance is passed in at call time
# to avoid circular imports.
from modules.simulator import GridWorld, Direction


# ── Color palette (normalized 0–1 for Matplotlib) ────────────────────────────

COLORS = {
    "background": (0.97, 0.97, 0.95),   # warm off-white
    "empty":      (1.00, 1.00, 1.00),   # white cell
    "grid_line":  (0.82, 0.82, 0.80),   # subtle grid
    "obstacle":   (0.25, 0.25, 0.28),   # near-black
    "robot":      (0.20, 0.75, 0.72),   # teal
    "robot_dark": (0.08, 0.50, 0.48),   # darker teal for arrow
    "text_light": (1.00, 1.00, 1.00),
    "text_dark":  (0.15, 0.15, 0.15),
}

# Direction → (dx, dy) arrow deltas (in cell-fraction units)
ARROW_DELTA = {
    Direction.UP:    (0,   0.28),
    Direction.DOWN:  (0,  -0.28),
    Direction.LEFT:  (-0.28, 0),
    Direction.RIGHT: (0.28,  0),
}


class GridVisualizer:
    """
    Renders a GridWorld to a Matplotlib figure.

    Usage:
        viz = GridVisualizer(cell_size=60)
        img_array = viz.render(world)
        viz.save(world, path="assets/grid_render.png")
    """

    def __init__(self, cell_size: int = 60, padding: int = 20):
        """
        Args:
            cell_size: Pixel size of each grid cell (before DPI scaling).
            padding:   Pixel padding around the grid.
        """
        self.cell_size = cell_size
        self.padding   = padding
        self.dpi       = 100

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, world: GridWorld) -> np.ndarray:
        """
        Render the world and return as an RGB numpy array (H, W, 3).
        Does NOT save to disk.
        """
        fig, ax = self._make_figure(world)
        self._draw(ax, world)
        img = self._fig_to_array(fig)
        plt.close(fig)
        return img

    def save(
        self,
        world: GridWorld,
        path: str = "assets/grid_render.png",
        show_title: bool = True,
    ) -> str:
        """
        Render the world and save to disk. Returns the absolute path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = self._make_figure(world)
        self._draw(ax, world, show_title=show_title)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor=COLORS["background"])
        plt.close(fig)
        return os.path.abspath(path)

    def show(self, world: GridWorld) -> None:
        """
        Render and open an interactive Matplotlib window.
        Only useful for local development (not Streamlit/server).
        """
        matplotlib.use("TkAgg")    # Switch to interactive backend
        fig, ax = self._make_figure(world)
        self._draw(ax, world, show_title=True)
        plt.show()
        plt.close(fig)

    # ── Figure setup ───────────────────────────────────────────────────────────

    def _make_figure(self, world: GridWorld):
        inch_size = self.cell_size / self.dpi
        w_in = world.width  * inch_size + (2 * self.padding / self.dpi)
        h_in = world.height * inch_size + (2 * self.padding / self.dpi) + 0.8  # legend row

        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=self.dpi)
        fig.patch.set_facecolor(COLORS["background"])
        ax.set_facecolor(COLORS["background"])

        # Axis limits: y is flipped so (0,0) is top-left
        ax.set_xlim(-0.5, world.width  - 0.5)
        ax.set_ylim( world.height - 0.5, -0.5)   # inverted y-axis
        ax.set_aspect("equal")

        # Clean up axes
        ax.set_xticks(range(world.width))
        ax.set_yticks(range(world.height))
        ax.set_xticklabels(range(world.width), fontsize=7, color="#888")
        ax.set_yticklabels(range(world.height), fontsize=7, color="#888")
        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return fig, ax

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self, ax, world: GridWorld, show_title: bool = True) -> None:
        self._draw_grid(ax, world)
        self._draw_obstacles(ax, world)
        self._draw_objects(ax, world)
        self._draw_robot(ax, world)
        self._draw_legend(ax, world)
        if show_title:
            self._draw_title(ax, world)

    def _draw_grid(self, ax, world: GridWorld) -> None:
        """Draw empty cell backgrounds and grid lines."""
        for y in range(world.height):
            for x in range(world.width):
                rect = mpatches.FancyBboxPatch(
                    (x - 0.48, y - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    facecolor=COLORS["empty"],
                    edgecolor=COLORS["grid_line"],
                    linewidth=0.5,
                )
                ax.add_patch(rect)

    def _draw_obstacles(self, ax, world: GridWorld) -> None:
        for (ox, oy) in world.obstacles:
            rect = mpatches.FancyBboxPatch(
                (ox - 0.48, oy - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["obstacle"],
                edgecolor=(0.15, 0.15, 0.18),
                linewidth=0.5,
            )
            ax.add_patch(rect)
            # Subtle "X" pattern to signal impassable
            ax.plot(
                [ox - 0.32, ox + 0.32], [oy - 0.32, oy + 0.32],
                color=(0.45, 0.45, 0.48), linewidth=0.8, zorder=2
            )
            ax.plot(
                [ox - 0.32, ox + 0.32], [oy + 0.32, oy - 0.32],
                color=(0.45, 0.45, 0.48), linewidth=0.8, zorder=2
            )

    def _draw_objects(self, ax, world: GridWorld) -> None:
        for name, obj in world.objects.items():
            # Normalize 0-255 RGB to 0-1
            color = tuple(c / 255 for c in obj.color)
            # Slightly lighter fill
            fill = tuple(min(1.0, c + 0.15) for c in color)

            rect = mpatches.FancyBboxPatch(
                (obj.x - 0.42, obj.y - 0.42), 0.84, 0.84,
                boxstyle="round,pad=0.06",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(rect)

            # Symbol in the center
            ax.text(
                obj.x, obj.y - 0.06,
                obj.symbol,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=COLORS["text_dark"],
                zorder=4,
            )
            # Name label below the cell
            ax.text(
                obj.x, obj.y + 0.38,
                name.replace("_", " "),
                ha="center", va="top",
                fontsize=5.5,
                color=COLORS["text_dark"],
                zorder=4,
            )

    def _draw_robot(self, ax, world: GridWorld) -> None:
        rx, ry = world.robot_x, world.robot_y

        # Robot body — teal circle
        circle = plt.Circle(
            (rx, ry), 0.36,
            facecolor=COLORS["robot"],
            edgecolor=COLORS["robot_dark"],
            linewidth=1.5,
            zorder=5,
        )
        ax.add_patch(circle)

        # "R" label
        ax.text(
            rx, ry - 0.04,
            "R",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=COLORS["text_light"],
            zorder=6,
        )

        # Directional arrow showing facing direction
        ddx, ddy = ARROW_DELTA[world.robot_facing]
        # Arrow starts from circle edge, points outward
        start_x = rx + ddx * 0.5
        start_y = ry + ddy * 0.5
        arrow = FancyArrow(
            start_x, start_y,
            ddx * 0.5, ddy * 0.5,
            width=0.06,
            head_width=0.14,
            head_length=0.10,
            length_includes_head=True,
            facecolor=COLORS["robot_dark"],
            edgecolor="none",
            zorder=7,
        )
        ax.add_patch(arrow)

    def _draw_legend(self, ax, world: GridWorld) -> None:
        """Compact inline legend as text below the grid."""
        legend_y = world.height - 0.1    # just below bottom row (y increases downward in data space)

        items = [
            (COLORS["robot"],    "Robot"),
            (COLORS["obstacle"], "Obstacle"),
        ] + [
            (tuple(c / 255 for c in obj.color), obj.name.replace("_", " "))
            for obj in world.objects.values()
        ]

        x_start = -0.4
        x_step  = 1.9
        for i, (color, label) in enumerate(items):
            cx = x_start + i * x_step
            dot = plt.Circle(
                (cx, legend_y + 0.55), 0.12,
                facecolor=color, edgecolor=(0.5, 0.5, 0.5), linewidth=0.5,
                transform=ax.transData, zorder=8,
                clip_on=False,
            )
            ax.add_patch(dot)
            ax.text(
                cx + 0.22, legend_y + 0.55,
                label,
                va="center", fontsize=6, color="#555",
                zorder=9,
            )

    def _draw_title(self, ax, world: GridWorld) -> None:
        ax.set_title(
            f"Embodied AI — Grid World   |   step {world.steps}   "
            f"|   robot @ ({world.robot_x}, {world.robot_y})   "
            f"|   facing {world.robot_facing.value}",
            fontsize=8,
            color="#444",
            pad=6,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_array(fig) -> np.ndarray:
        """Convert a Matplotlib figure to an (H, W, 3) uint8 RGB array.
        Supports both old (tostring_rgb) and new (buffer_rgba) Matplotlib APIs.
        """
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        try:
            # Matplotlib >= 3.8
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            return buf.reshape(h, w, 4)[:, :, :3]   # drop alpha channel
        except AttributeError:
            # Matplotlib < 3.8
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
