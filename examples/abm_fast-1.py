"""
Agent-Based Model: Growing and Dividing Agents
================================================
Each agent has a position (x, y), an area, and a constant growth rate.
When an agent's area exceeds a threshold, it divides into two daughter agents.

NumPy-accelerated version with fixed-size arrays (MAX_AGENTS cap).
The O(N²) velocity kernel is now fully vectorised.
"""

import math
import random
import numpy as np
import uuid
import csv
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAX_AGENTS = 400          # hard cap; determines pre-allocated array size
# MAX_AGENTS = 100          # hard cap; determines pre-allocated array size
max_ID = 0


# ---------------------------------------------------------------------------
# Agent dataclass (kept for visualisation / history compatibility)
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    x: float
    y: float
    vel_x: float
    vel_y: float
    area: float = 78.5
    growth_rate: float = 0.1778
    division_area: float = 157.08
    norm_rand: float = 2.0
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    ID: int = 0
    time_step: int = 0

    def grow(self) -> None:
        self.area += self.growth_rate
        self.time_step += 1

    def should_divide(self) -> bool:
        return self.area >= self.division_area

    def divide(self, separation: float = 0.5, noise: float = 0.1):
        global max_ID
        daughter_area   = self.area / 2
        daughter_radius = math.sqrt(daughter_area / math.pi)

        max_ID += 1

        # initially trying what I thought was closer to PhysiCell, but it seemed to generate biased/skewed division
        # theta = random.uniform(0, 1)
        # xv = math.cos(theta) * 2 * daughter_radius
        # yv = math.sin(theta) * 2 * daughter_radius

        # this results in non-biased monolayer growth about the origin
        theta = random.random() * 2 * math.pi
        xv = math.cos(theta) * daughter_radius
        yv = math.sin(theta) * daughter_radius

        def _norm_rand():
            v = random.normalvariate(2.0, 0.4)
            while v < 0:
                v = random.normalvariate(2.0, 0.4)
            return v

        nr1 = _norm_rand()
        nr2 = _norm_rand()

        d1 = Agent(x=self.x,      y=self.y,      vel_x=0, vel_y=0,
                   area=daughter_area, growth_rate=self.growth_rate,
                   division_area=78.54 * nr1, norm_rand=nr1, ID=self.ID)
        d2 = Agent(x=self.x + xv, y=self.y + yv, vel_x=0, vel_y=0,
                   area=daughter_area, growth_rate=self.growth_rate,
                   division_area=78.54 * nr2, norm_rand=nr2, ID=max_ID)
        return d1, d2


# ---------------------------------------------------------------------------
# Numpy-backed simulation state
# ---------------------------------------------------------------------------
class _NpState:
    """
    Fixed-size numpy arrays that mirror the live agent population.
    Slot 0..n-1 are active; slots n..MAX_AGENTS-1 are unused.
    """
    __slots__ = ("x", "y", "vx", "vy", "area", "growth_rate",
                 "division_area", "n")

    def __init__(self, cap: int = MAX_AGENTS):
        self.x             = np.zeros(cap, dtype=np.float64)
        self.y             = np.zeros(cap, dtype=np.float64)
        self.vx            = np.zeros(cap, dtype=np.float64)
        self.vy            = np.zeros(cap, dtype=np.float64)
        self.area          = np.zeros(cap, dtype=np.float64)
        self.growth_rate   = np.zeros(cap, dtype=np.float64)
        self.division_area = np.zeros(cap, dtype=np.float64)
        self.n             = 0                   # number of active agents

    def load_from_agents(self, agents: list):
        n = len(agents)
        self.n = n
        for i, a in enumerate(agents):
            self.x[i]             = a.x
            self.y[i]             = a.y
            self.vx[i]            = a.vel_x
            self.vy[i]            = a.vel_y
            self.area[i]          = a.area
            self.growth_rate[i]   = a.growth_rate
            self.division_area[i] = a.division_area

    def write_back_to_agents(self, agents: list):
        for i, a in enumerate(agents):
            a.x     = self.x[i]
            a.y     = self.y[i]
            a.vel_x = self.vx[i]
            a.vel_y = self.vy[i]
            a.area  = self.area[i]

    def add_agent(self, a: "Agent"):
        i = self.n
        self.x[i]             = a.x
        self.y[i]             = a.y
        self.vx[i]            = a.vel_x
        self.vy[i]            = a.vel_y
        self.area[i]          = a.area
        self.growth_rate[i]   = a.growth_rate
        self.division_area[i] = a.division_area
        self.n += 1

    def remove_index(self, idx: int):
        """Swap-remove: replace slot idx with the last slot, decrement n."""
        last = self.n - 1
        if idx != last:
            for arr in (self.x, self.y, self.vx, self.vy,
                        self.area, self.growth_rate, self.division_area):
                arr[idx] = arr[last]
        self.n -= 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class Simulation:
    """
    Manages a population of Agents over discrete time steps.
    The grow/divide loop uses Python objects; the O(N²) velocity kernel
    and position update are fully vectorised with NumPy.
    """

    def __init__(
        self,
        n_seed: int = 1,
        growth_rate: float = 0.0,
        division_area: float = 0.0,
        norm_rand: float = 0.0,
        initial_area: float = 0.0,
    ):
        self.agents: list[Agent] = [
            Agent(
                x=0, y=0, vel_x=0, vel_y=0,
                area=initial_area,
                growth_rate=growth_rate,
                division_area=division_area,
                norm_rand=norm_rand,
            )
            for _ in range(n_seed)
        ]
        self._np  = _NpState(MAX_AGENTS)
        self._np.load_from_agents(self.agents)

        self.time: int = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_np_from_agents(self):
        self._np.load_from_agents(self.agents)

    def _sync_agents_from_np(self):
        self._np.write_back_to_agents(self.agents)

    # ------------------------------------------------------------------
    # Vectorised velocity update  (O(N²) → numpy broadcast)
    # ------------------------------------------------------------------
    def _update_velocities_np(self, cell_cell_repulsion_strength: float = 10.0):
        s  = self._np
        n  = s.n
        if n <= 1:
            s.vx[:n] = 0.0
            s.vy[:n] = 0.0
            return

        x  = s.x[:n]
        y  = s.y[:n]
        r  = np.sqrt(s.area[:n] / math.pi)   # radii  (n,)

        # Pairwise displacements  (n, n)
        dx   = x[:, None] - x[None, :]       # (n, n)
        dy   = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)    # (n, n)

        # Sum of radii  (n, n)
        R    = r[:, None] + r[None, :]

        # Interaction mask: dist < 1.25 * R  AND  i != j
        np.fill_diagonal(dist, np.inf)       # exclude self
        # mask = dist < 1.25 * R              # (n, n)
        # mask = dist < 20.0 * R              # (n, n)
        mask = dist < R              # (n, n)

        # Repulsion magnitude  (1 - d/R)^2 * strength / d
        # where temp_r = 0 when dist >= R
        overlap = np.where(dist < R, 1.0 - dist / R, 0.0)   # (n, n)
        temp_r  = np.where(mask, overlap * overlap * cell_cell_repulsion_strength / dist, 0.0)

        # Velocity accumulation
        s.vx[:n] = (temp_r * dx).sum(axis=1)
        s.vy[:n] = (temp_r * dy).sum(axis=1)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self) -> int:
        divisions = 0

        # --- 1) Grow + divide (Python loop; N is small) ---
        survivors: list[Agent] = []
        for agent in self.agents:
            agent.grow()
            if agent.should_divide():
                d1, d2 = agent.divide()
                survivors.extend([d1, d2])
                divisions += 1
            else:
                survivors.append(agent)
        self.agents = survivors

        # --- 2) Sync numpy arrays ---
        self._np.load_from_agents(self.agents)

        # --- 3) Vectorised velocity update ---
        self._update_velocities_np()

        # --- 4) Vectorised position update ---
        n = self._np.n
        self._np.x[:n] += self._np.vx[:n]
        self._np.y[:n] += self._np.vy[:n]

        # --- 5) Write positions / velocities back to Agent objects ---
        self._np.write_back_to_agents(self.agents)

        self.time += 1
        self.history.append({
            "time":       self.time,
            "population": len(self.agents),
            "divisions":  divisions,
            "mean_area":  float(self._np.area[:n].mean()),
        })

        return divisions

    # ------------------------------------------------------------------
    # No longer called??
    def run(self, steps: int, max_agents: int = MAX_AGENTS) -> None:
        for _ in range(steps):
            self.step()
            if len(self.agents) >= max_agents:
                print(f"  Stopped early at t={self.time}: {len(self.agents)} agents.")
                break


    def precompute(self, steps: int, max_agents: int = MAX_AGENTS) -> None:
        import copy
        print(f"Pre-computing {steps} steps...", end=" ", flush=True)
        self._frames: list[list[Agent]] = [copy.deepcopy(self.agents)]

        for _ in range(steps):
            self.step()
            self._frames.append(copy.deepcopy(self.agents))
            if len(self.agents) >= max_agents:
                print(f"(stopped early at t={self.time} — {len(self.agents)} agents)", end=" ")
                break

        print(f"done. {len(self._frames)} frames stored.")

        # save final results in .csv
        file_out = f'py_monolayer.csv'
        print("--> ",file_out)
        with open(file_out, "w", newline="") as file:
            writer = csv.writer(file)

            # Write each row: x,y,g,n  (where g=growing (0/1), n=# of nbrs)
            # writer.writerow(['x_pos','y_pos','radius_i','f_i','a_i'])
            writer.writerow(['x_pos','y_pos','radius_i','norm_rand_i'])
            for agent in self.agents:
                # writer.writerow([x_pos[jdx],y_pos[jdx],radius_i[jdx],f_i[jdx],a_i[jdx]])
                radius = math.sqrt(agent.area / math.pi)
                writer.writerow([agent.x,agent.y,radius,agent.norm_rand])

    # ------------------------------------------------------------------
    def interactive_viewer(self, title: str = "Agent-Based Model", interval: int = 100):
        """
        Interactive matplotlib viewer (unchanged from original).
        """
        import copy
        from matplotlib.widgets import Button, Slider
        from matplotlib.animation import FuncAnimation

        if not hasattr(self, "_frames"):
            raise RuntimeError("Call precompute() before interactive_viewer().")

        frames   = self._frames
        n_frames = len(frames)
        cmap     = plt.cm.plasma

        fig = plt.figure(figsize=(12, 6), facecolor="#0d0d0d")
        fig.suptitle(title, fontsize=13, fontweight="bold", color="white")

        ax_space = fig.add_axes([0.04, 0.22, 0.46, 0.70])
        ax_pop   = fig.add_axes([0.56, 0.22, 0.40, 0.70])

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

        all_times = list(range(n_frames))
        all_pops  = [len(f) for f in frames]
        ax_pop.plot(all_times, all_pops, color="#334455", linewidth=1.5, zorder=1)
        pop_line,   = ax_pop.plot([], [], color="#00d4ff", linewidth=2, zorder=2)
        time_marker = ax_pop.axvline(x=0, color="#ff6b6b", linewidth=1.5,
                                     linestyle="--", zorder=3)
        ax_pop.set_xlim(0, n_frames - 1)
        ax_pop.set_ylim(0, max(all_pops) * 1.15 + 1)

        win_size = 150
        ax_space.set_xlim(-win_size, win_size)
        ax_space.set_ylim(-win_size, win_size)
        title_text = ax_space.set_title("", color="white", fontsize=10)

        btn_color = "#1e2d3d"
        btn_hover = "#2e4d6d"

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

        state = {"frame": 0, "playing": False}

        def _draw_frame(idx: int) -> None:
            idx = max(0, min(idx, n_frames - 1))
            state["frame"] = idx
            agent_list = frames[idx]

            for p in list(ax_space.patches):
                p.remove()
            for agent in agent_list:
                ax_space.add_patch(patches.Circle(
                    (agent.x, agent.y),
                    radius=math.sqrt(agent.area / math.pi),
                    facecolor='white', alpha=0.9,
                    linewidth=1.0, edgecolor="darkgray",
                ))

            title_text.set_text(f"t = {idx}   |   agents = {len(agent_list)}")
            pop_line.set_data(all_times[:idx + 1], all_pops[:idx + 1])
            time_marker.set_xdata([idx, idx])

            slider.eventson = False
            slider.set_val(idx)
            slider.eventson = True
            fig.canvas.draw_idle()

        def on_back(_e):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(state["frame"] - 1)

        def on_fwd(_e):
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _draw_frame(state["frame"] + 1)

        def on_play(_e):
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

        def _tick(_frame):
            if state["playing"]:
                nxt = state["frame"] + 1
                if nxt >= n_frames:
                    state["playing"] = False
                    btn_play.label.set_text("▶ Play")
                else:
                    _draw_frame(nxt)

        anim = FuncAnimation(fig, _tick, interval=interval, cache_frame_data=False)
        _draw_frame(0)
        plt.show()
        return anim


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)

    sim = Simulation(
        n_seed=1,
        growth_rate=0.1778,
        division_area=157.08,
        norm_rand=2.0,
        initial_area=78.54,
    )

    sim.precompute(steps=50000, max_agents=MAX_AGENTS)

    sim.interactive_viewer(
        title="Growing Monolayer",
        interval=1,
    )
