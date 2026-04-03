"""
Agent-Based Model: Growing and Dividing Agents
================================================
Each agent has a position (x, y), a radius, and a constant growth rate.
When an agent's radius exceeds a threshold, it divides into two daughter agents.
"""

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Agent:
    """
    An agent with a 2D position, radius, and constant growth rate.

    Attributes:
        x, y          : Position in 2D space.
        radius        : Current radius of the agent.
        growth_rate   : Rate at which the radius increases per time step (units/step).
        division_radius: Radius at which the agent divides.
        agent_id      : Unique identifier.
        parent_id     : ID of the parent agent (None for seed agents).
        generation    : Number of divisions from the original ancestor.
    """
    x: float
    y: float
    radius: float = 1.0
    growth_rate: float = 0.05
    division_radius: float = 2.0
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    generation: int = 0

    def grow(self) -> None:
        """Increase radius by growth_rate for one time step."""
        self.radius += self.growth_rate

    def should_divide(self) -> bool:
        """Return True if this agent has reached or exceeded its division radius."""
        return self.radius >= self.division_radius

    def divide(self, separation: float = 0.5, noise: float = 0.1) -> tuple["Agent", "Agent"]:
        """
        Divide this agent into two daughter agents.

        The daughters are placed symmetrically around the parent's position,
        each with half the parent's volume (radius scaled by 1/cbrt(2)).

        Args:
            separation: Extra offset distance for daughter placement beyond
                        their initial radius (prevents immediate overlap).
            noise:      Random angular perturbation (radians) so divisions
                        aren't always perfectly axis-aligned.

        Returns:
            A tuple of two new Agent instances (daughter_1, daughter_2).
        """
        # Daughter radius: conserve volume in 2D (area), so r_d = r / sqrt(2)
        daughter_radius = self.radius / math.sqrt(2)

        # Random division axis with optional noise
        angle = random.uniform(0, 2 * math.pi)

        # Offset each daughter by daughter_radius + separation along the axis
        offset = daughter_radius + separation
        dx = offset * math.cos(angle)
        dy = offset * math.sin(angle)

        daughter_kwargs = dict(
            radius=daughter_radius,
            growth_rate=self.growth_rate,
            division_radius=self.division_radius,
            parent_id=self.agent_id,
            generation=self.generation + 1,
        )

        daughter_1 = Agent(x=self.x + dx, y=self.y + dy, **daughter_kwargs)
        daughter_2 = Agent(x=self.x - dx, y=self.y - dy, **daughter_kwargs)

        return daughter_1, daughter_2


class Simulation:
    """
    Manages a population of Agents over discrete time steps.

    At each step:
      1. Every agent grows.
      2. Agents that exceed division_radius are replaced by two daughters.
    """

    def __init__(
        self,
        n_seed: int = 1,
        growth_rate: float = 0.05,
        division_radius: float = 2.0,
        initial_radius: float = 1.0,
        spawn_range: float = 5.0,
    ):
        self.agents: list[Agent] = [
            Agent(
                x=random.uniform(-spawn_range, spawn_range),
                y=random.uniform(-spawn_range, spawn_range),
                radius=initial_radius,
                growth_rate=growth_rate,
                division_radius=division_radius,
            )
            for _ in range(n_seed)
        ]
        self.time: int = 0
        self.history: list[dict] = []

    def step(self) -> int:
        """
        Advance the simulation by one time step.

        Returns:
            Number of division events that occurred this step.
        """
        survivors: list[Agent] = []
        divisions = 0

        for agent in self.agents:
            agent.grow()
            if agent.should_divide():
                d1, d2 = agent.divide()
                survivors.extend([d1, d2])
                divisions += 1
            else:
                survivors.append(agent)

        self.agents = survivors
        self.time += 1

        self.history.append({
            "time": self.time,
            "population": len(self.agents),
            "divisions": divisions,
            "mean_radius": sum(a.radius for a in self.agents) / len(self.agents),
        })

        return divisions

    def run(self, steps: int, max_agents: int = 500) -> None:
        """
        Run the simulation for a given number of steps or until max_agents is reached.

        Args:
            steps:      Maximum number of time steps to run.
            max_agents: Stop early if population exceeds this threshold.
        """
        for _ in range(steps):
            self.step()
            if len(self.agents) >= max_agents:
                print(f"  Stopped early at t={self.time}: {len(self.agents)} agents.")
                break

    def precompute(self, steps: int, max_agents: int = 500) -> None:
        """
        Run the full simulation up-front and store every frame's agent list.
        Required before calling interactive_viewer().

        Args:
            steps:      Number of time steps to pre-compute.
            max_agents: Stop early if population exceeds this.
        """
        import copy
        print(f"Pre-computing {steps} steps...", end=" ", flush=True)
        # Frame 0 = initial state (before any stepping)
        self._frames: list[list[Agent]] = [copy.deepcopy(self.agents)]

        for _ in range(steps):
            self.step()
            self._frames.append(copy.deepcopy(self.agents))
            if len(self.agents) >= max_agents:
                print(f"(stopped early at t={self.time} — {len(self.agents)} agents)", end=" ")
                break

        print(f"done. {len(self._frames)} frames stored.")

    def interactive_viewer(self, title: str = "Agent-Based Model", interval: int = 100) -> None:
        """
        Open an interactive matplotlib window with:
          ◀◀  Step back one frame
          ▶▶  Step forward one frame
          ▶ / ■  Play / Pause the full simulation
          A time-step slider for direct scrubbing

        Requires precompute() to have been called first.
        """
        import copy
        from matplotlib.widgets import Button, Slider
        from matplotlib.animation import FuncAnimation

        if not hasattr(self, "_frames"):
            raise RuntimeError("Call precompute() before interactive_viewer().")

        frames     = self._frames
        n_frames   = len(frames)
        cmap       = plt.cm.plasma

        # ------------------------------------------------------------------ #
        # Layout: plots occupy the top 80 %, widgets the bottom 20 %
        # ------------------------------------------------------------------ #
        fig = plt.figure(figsize=(15, 8), facecolor="#0d0d0d")
        fig.suptitle(title, fontsize=13, fontweight="bold", color="white")

        # Two data axes
        ax_space = fig.add_axes([0.04, 0.22, 0.46, 0.70])   # left: spatial
        ax_pop   = fig.add_axes([0.56, 0.22, 0.40, 0.70])   # right: population

        for ax in (ax_space, ax_pop):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")

        ax_space.set_aspect("equal")
        ax_space.set_xlabel("x", color="white")
        ax_space.set_ylabel("y", color="white")
        ax_pop.set_xlabel("Time step", color="white")
        ax_pop.set_ylabel("Agents", color="white")
        ax_pop.set_title("Population Over Time", color="white")
        ax_pop.grid(True, alpha=0.2, color="white")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 10))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_space, fraction=0.046, pad=0.04)
        cbar.set_label("Generation", color="white")
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

        # Population line (full history, dimmed) + current-position marker
        all_times = list(range(n_frames))
        all_pops  = [len(f) for f in frames]
        ax_pop.plot(all_times, all_pops, color="#334455", linewidth=1.5, zorder=1)
        pop_line,   = ax_pop.plot([], [], color="#00d4ff", linewidth=2, zorder=2)
        time_marker = ax_pop.axvline(x=0, color="#ff6b6b", linewidth=1.5,
                                     linestyle="--", zorder=3)
        ax_pop.set_xlim(0, n_frames - 1)
        ax_pop.set_ylim(0, max(all_pops) * 1.15 + 1)

        # Stable world bounds (union of all frame extents)
        all_x = [a.x for f in frames for a in f]
        all_y = [a.y for f in frames for a in f]
        max_r = max((a.radius for f in frames for a in f), default=1)
        pad   = max_r * 2
        ax_space.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax_space.set_ylim(min(all_y) - pad, max(all_y) + pad)

        title_text = ax_space.set_title("", color="white", fontsize=10)

        # ------------------------------------------------------------------ #
        # Widget axes
        # ------------------------------------------------------------------ #
        btn_color      = "#1e2d3d"
        btn_hover      = "#2e4d6d"

        ax_slider = fig.add_axes([0.10, 0.10, 0.80, 0.03], facecolor="#1a1a2e")
        ax_back   = fig.add_axes([0.28, 0.02, 0.08, 0.055], facecolor=btn_color)
        ax_play   = fig.add_axes([0.38, 0.02, 0.10, 0.055], facecolor=btn_color)
        ax_fwd    = fig.add_axes([0.50, 0.02, 0.08, 0.055], facecolor=btn_color)

        slider   = Slider(ax_slider, "t", 0, n_frames - 1,
                          valinit=0, valstep=1, color="#00d4ff")
        slider.label.set_color("white")
        slider.valtext.set_color("white")

        btn_back = Button(ax_back, "◀◀", color=btn_color, hovercolor=btn_hover)
        btn_play = Button(ax_play, "▶ Play", color=btn_color, hovercolor=btn_hover)
        btn_fwd  = Button(ax_fwd,  "▶▶", color=btn_color, hovercolor=btn_hover)

        for btn in (btn_back, btn_play, btn_fwd):
            btn.label.set_color("white")
            btn.label.set_fontsize(10)

        # ------------------------------------------------------------------ #
        # Draw helpers
        # ------------------------------------------------------------------ #
        state = {"frame": 0, "playing": False}

        def _draw_frame(idx: int) -> None:
            idx = max(0, min(idx, n_frames - 1))
            state["frame"] = idx

            agent_list = frames[idx]
            max_gen    = max((a.generation for a in agent_list), default=0) or 1

            for p in list(ax_space.patches):
                p.remove()
            for agent in agent_list:
                color = cmap(agent.generation / max(max_gen, 10))
                ax_space.add_patch(patches.Circle(
                    (agent.x, agent.y), radius=agent.radius,
                    facecolor=color, alpha=0.72,
                    linewidth=0.4, edgecolor="white",
                ))

            title_text.set_text(
                f"t = {idx}   |   agents = {len(agent_list)}   |   max gen = {max_gen}"
            )

            pop_line.set_data(all_times[:idx + 1], all_pops[:idx + 1])
            time_marker.set_xdata([idx, idx])

            # Update slider without triggering its callback
            slider.eventson = False
            slider.set_val(idx)
            slider.eventson = True

            fig.canvas.draw_idle()

        # ------------------------------------------------------------------ #
        # Button / slider callbacks
        # ------------------------------------------------------------------ #
        def on_back(_event):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(state["frame"] - 1)

        def on_fwd(_event):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(state["frame"] + 1)

        def on_play(_event):
            state["playing"] = not state["playing"]
            btn_play.label.set_text("■ Pause" if state["playing"] else "▶ Play")
            fig.canvas.draw_idle()

        def on_slider(val):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(int(val))

        btn_back.on_clicked(on_back)
        btn_fwd.on_clicked(on_fwd)
        btn_play.on_clicked(on_play)
        slider.on_changed(on_slider)

        # ------------------------------------------------------------------ #
        # Timer-driven playback
        # ------------------------------------------------------------------ #
        def _tick(_frame):
            if state["playing"]:
                nxt = state["frame"] + 1
                if nxt >= n_frames:
                    state["playing"] = False
                    btn_play.label.set_text("▶ Play")
                else:
                    _draw_frame(nxt)

        anim = FuncAnimation(fig, _tick, interval=interval, cache_frame_data=False)

        # Draw initial frame
        _draw_frame(0)
        plt.show()

        return anim  # keep reference alive so GC doesn't kill the timer


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation  # ensure import available

    random.seed(42)

    sim = Simulation(
        n_seed=3,            # Start with 3 agents
        growth_rate=0.08,    # Radius grows by 0.08 per step
        division_radius=2.0, # Agent divides when radius >= 2.0
        initial_radius=1.0,
        spawn_range=10.0,
    )

    # 1. Pre-compute all frames (enables backward scrubbing)
    sim.precompute(steps=80, max_agents=300)

    # 2. Open the interactive widget viewer
    sim.interactive_viewer(
        title="Growing & Dividing Agent-Based Model",
        interval=120,   # ms between auto-play frames
    )
