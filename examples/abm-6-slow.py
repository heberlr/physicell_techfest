"""
Agent-Based Model: Growing and Dividing Agents
================================================
Each agent has a position (x, y), a area, and a constant growth rate.
When an agent's area exceeds a threshold, it divides into two daughter agents.
"""

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

max_ID = 0

@dataclass
class Agent:
    """
    An agent with a 2D position, area, and constant growth rate.

    Attributes:
        x, y          : Position 
        vel_x, vel_y  : Velocity 
        area        : Current area of the agent.
        growth_rate   : Rate at which the area increases per time step (units/step).
        division_area : area at which the agent divides.
        norm_rand     : X ∼ N(2, 0.4^2), truncated such that X > 0, redraw X otherwise
        agent_id      : Unique identifier.
        parent_id     : ID of the parent agent (None for seed agents).
        generation    : Number of divisions from the original ancestor.
    """
    x: float
    y: float
    vel_x: float
    vel_y: float
    area: float = 78.5  # pi * 5^2
    growth_rate: float = 0.1778   # microns^2(?)/min  0.05
    division_area: float = 157.08        # division area=7.07
    norm_rand: float = 2.0
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    ID: int = 0
    time_step: int = 0

    def grow(self) -> None:
        """Increase area by growth_rate for one time step."""
        self.area += self.growth_rate
        self.time_step += 1
        # if self.time_step % 5 == 0:
        if False: # self.ID == 0:
            print(f"grow(): ID=0, time_step={self.time_step}, {self.agent_id}: area= {self.area}")

    def should_divide(self) -> bool:
        """Return True if this agent has reached or exceeded its division area."""
        return self.area >= self.division_area

    def divide(self, separation: float = 0.5, noise: float = 0.1) -> tuple["Agent", "Agent"]:
        global max_ID
        """
        Divide this agent into two daughter agents.

        The daughters are placed symmetrically around the parent's position,
        each with half the parent's volume (area scaled by 1/cbrt(2)).

        Args:
            separation: Extra offset distance for daughter placement beyond
                        their initial area (prevents immediate overlap).
            noise:      Random angular perturbation (radians) so divisions
                        aren't always perfectly axis-aligned.

        Returns:
            A tuple of two new Agent instances (daughter_1, daughter_2).
        """
        # Daughter area: conserve volume in 2D (area), so r_d = r / sqrt(2)
        # daughter_area = self.area / math.sqrt(2)
        daughter_area = self.area / 2
        daughter_radius = math.sqrt(daughter_area/math.pi)
        # print(f"-- divide(): ID={self.ID}, separation={separation}, daughter_area={daughter_area } ")
        # print(f"-- divide(): ID={self.ID}, daughter_area={daughter_area}, daughter_radius={daughter_radius}   ")

        # Random division axis with optional noise
        # angle = random.uniform(0, 2 * math.pi)

        # In PhysiCell _utilities() UniformOnUnitSphere: get random vector, 
        # double z = UniformRandom();
        # z *= 2.0; 
        # z -= 1.0; // Choose z uniformly distributed in [-1,1].
        #
        # theta = UniformRandomm();
        # theta *= two_pi;
        # 

        max_ID += 1
        # print("   max_ID=",max_ID)
        theta = random.uniform(0,1)
        # if max_ID == 1:
        #     theta = 0  # rwh: dummy test
        # if max_ID >= 2:
        #     # theta = 0.7854
        #     theta = 1.5708
        # theta *= 6.283185307179
        # theta = 0.

        # Offset each daughter by daughter_area + separation along the axis
        # radius = math.sqrt(daughter_area/math.pi)
        # offset = radius + separation
        # dx = offset * math.cos(angle)
        # dy = offset * math.sin(angle)

        radius = math.sqrt(self.area/math.pi)
        # print(f"  self.area= {self.area}, --> radius= {radius}")

        xv = math.cos(theta)
        yv = math.sin(theta)
        # print("  xv,yv= ",xv,yv)

        # this makes it so the daughter cells overlap
        # radius = math.sqrt(daughter_area/math.pi)
        # xv *= radius
        # yv *= radius

        # this makes it so the cells are kissing (just touching)
        xv *= 2*daughter_radius
        yv *= 2*daughter_radius
        # print("  r-scaled xv,yv= ",xv,yv)

        norm_rand1 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4
        while norm_rand1 < 0:
             norm_rand1 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4

        daughter1_kwargs = dict(
            area=daughter_area,
            growth_rate=self.growth_rate,
            division_area=78.54 * norm_rand1,
            norm_rand=norm_rand1,
            ID=self.ID,
        )

        # 2nd daughter cell has independent division_area
        norm_rand2 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4
        while norm_rand2 < 0:
             norm_rand2 = random.normalvariate(mu=2.0, sigma=0.4)   # mean=2, stddev=0.4

        daughter2_kwargs = dict(
            area=daughter_area,
            growth_rate=self.growth_rate,
            division_area=78.54 * norm_rand2,
            norm_rand=norm_rand2,
            ID=max_ID,
        )

        # daughter_1 = Agent(x=self.x + dx, y=self.y + dy, **daughter_kwargs)
        # daughter_1 = Agent(x=self.x, y=self.y, **daughter_kwargs)
        # daughter_1 = Agent(x=self.x, y=self.y, vel_x=0, vel_y=0, norm_rand=norm_rand1, **daughter1_kwargs)
        daughter_1 = Agent(x=self.x, y=self.y, vel_x=0, vel_y=0, **daughter1_kwargs)
        # daughter_1.ID = self.ID

        # daughter_2 = Agent(x=self.x - dx, y=self.y - dy, **daughter_kwargs)
        daughter_2 = Agent(x=self.x + xv, y=self.y + yv, vel_x=0, vel_y=0, **daughter2_kwargs)
        # daughter_2.ID = self.ID

        return daughter_1, daughter_2


class Simulation:
    """
    Manages a population of Agents over discrete time steps.

    At each step:
      1. Every agent grows.
      2. Agents that exceed division_area are replaced by two daughters.
    """

    def __init__(
        self,
        n_seed: int = 1,
        growth_rate: float = 0.05,
        division_area: float = 2.0,
        norm_rand: float = 2.0,
        initial_area: float = 1.0,
        spawn_range: float = 1.0,
    ):
        self.agents: list[Agent] = [
            Agent(
                # x=random.uniform(-spawn_range, spawn_range),
                # y=random.uniform(-spawn_range, spawn_range),
                x=0,
                y=0,
                vel_x=0,
                vel_y=0,
                area=initial_area,
                growth_rate=growth_rate,
                division_area=division_area,
                norm_rand=norm_rand,
            )
            for _ in range(n_seed)
        ]
        self.time: int = 0
        self.history: list[dict] = []

    def update_cell_velocity(self) -> None:
        return

    def step(self) -> int:
        """
        Advance the simulation by one time step.

        Returns:
            Number of division events that occurred this step.
        """
        survivors: list[Agent] = []
        divisions = 0

        # 1) Grow all agents
        for agent in self.agents:
            agent.grow()
            if agent.should_divide():
                d1, d2 = agent.divide()
                survivors.extend([d1, d2])
                divisions += 1
            else:
                survivors.append(agent)
            # print("survivors=",survivors)

        self.agents = survivors

        # void standard_update_cell_velocity( Cell* pCell, Phenotype& phenotype, double dt)
        # for(neighbor = pCell->get_container()->agent_grid[pCell->get_current_mechanics_voxel_index()].begin(); neighbor != end; ++neighbor)
	    # {
		#     pCell->add_potentials(*neighbor);
	    # }

        # 2) Update each agent's velocity vector:  like Cell::add_potentials(Cell* other_agent)
        cell_cell_repulsion_strength = 10.0
        for agent in self.agents:
            agent_r = math.sqrt(agent.area/math.pi)
            agent.vel_x = 0
            agent.vel_y = 0
            for agent2 in self.agents:
                if agent.agent_id != agent2.agent_id:
                    # displacement[i] = position[i] - (*other_agent).position[i]; 
                    xdel = agent.x - agent2.x   # "displacement" vector in C++
                    ydel = agent.y - agent2.y
                    dist = math.sqrt(xdel*xdel + ydel*ydel)
                    # distance = std::max(sqrt(distance), 0.00001); 

	# //Repulsive
	# double R = phenotype.geometry.radius+ (*other_agent).phenotype.geometry.radius; 

                    agent2_r = math.sqrt(agent2.area/math.pi)
                    if dist < 1.25 * (agent_r + agent2_r):
                        R = agent_r + agent2_r
                        if dist > R:
                            temp_r = 0
                        else:
                            # print(f"update velocity of cell {agent.agent_id} due to nbr {agent.agent_id}")
                            temp_r = -dist  # -d
                            temp_r /= R # -d/R
                            temp_r += 1.0 # 1-d/R
                            temp_r *= temp_r # (1-d/R)^2 
                            
                            # add the relative pressure contribution 
                            # state.simple_pressure += ( temp_r / simple_pressure_scale ); // New July 2017 

                        # double effective_repulsion = sqrt( phenotype.mechanics.cell_cell_repulsion_strength * other_agent->phenotype.mechanics.cell_cell_repulsion_strength ); 
                        # effective_repulsion = math.sqrt(cell_cell_repulsion_strength * cell_cell_repulsion_strength) 
                        # effective_repulsion = cell_cell_repulsion_strength  
                        # temp_r *= effective_repulsion 
                        temp_r *= cell_cell_repulsion_strength 

                        # skip over adhesion, there is none

                        # if( fabs(temp_r) < 1e-16 )
                        # { return; }
                        temp_r /= dist

                        # velocity[i] += displacement[i] * temp_r; 
	                    # axpy( &velocity , temp_r , displacement ); 
                        agent.vel_x += xdel * temp_r
                        agent.vel_y += ydel * temp_r
                        # if self.time >= 90 and self.time <=120:  #rwh
                            # print(f"--ID={agent.ID}, step={self.time}: update velocity for {agent.agent_id}: vel_x={agent.vel_x}, _y={agent.vel_y}")
                            # print(f"--step={self.time}: update velocity for {agent.ID}: vel_x={agent.vel_x}, _y={agent.vel_y}")

        # 3) Update each agent's position
        for agent in self.agents:
            agent.x += agent.vel_x
            agent.y += agent.vel_y


        # self.agents = survivors
        self.time += 1

        self.history.append({
            "time": self.time,
            "population": len(self.agents),
            "divisions": divisions,
            "mean_area": sum(a.area for a in self.agents) / len(self.agents),
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
        # fig = plt.figure(figsize=(15, 8), facecolor="#0d0d0d")
        fig = plt.figure(figsize=(12, 6), facecolor="#0d0d0d")
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
        # cbar = fig.colorbar(sm, ax=ax_space, fraction=0.046, pad=0.04)
        # cbar.set_label("Generation", color="white")
        # cbar.ax.yaxis.label.set_color("white")
        # cbar.ax.tick_params(colors="white")

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
        max_r = max((a.area for f in frames for a in f), default=1)  # now doing area, not radii
        pad   = max_r * 2
        pad   = 20
        # ax_space.set_xlim(min(all_x) - pad, max(all_x) + pad)
        # ax_space.set_ylim(min(all_y) - pad, max(all_y) + pad)
        win_size = 200  # rwh
        ax_space.set_xlim(-win_size, win_size)
        ax_space.set_ylim(-win_size, win_size)

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
            # max_gen    = max((a.generation for a in agent_list), default=0) or 1

            for p in list(ax_space.patches):
                p.remove()
            for agent in agent_list:
                # color = cmap(agent.generation / max(max_gen, 10))
                ax_space.add_patch(patches.Circle(
                    (agent.x, agent.y), 
                    radius=math.sqrt(agent.area / math.pi),
                    facecolor='gray', alpha=0.72,
                    linewidth=1.0, edgecolor="white",
                ))

            title_text.set_text(
                f"t = {idx}   |   agents = {len(agent_list)}"  #   |   max gen = {max_gen}"
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
        n_seed=1,           # Start with n agents
        # growth_rate=0.1778,   # micron^2 / min
        growth_rate=0.1778,   # 0.889 original;  5x slower=0.1778  (micron^2 / min)
        division_area=157.08,  #  vs. double area = 2 * 78.54 = 157.08
        norm_rand=2.0,  #  X ~ N(2, 0.4^2)
        initial_area=78.54,   #  area = pi * r^2 = pi * 25 = 78.54
        spawn_range=0.0,
    )

    # 1. Pre-compute all frames (enables backward scrubbing)
    sim.precompute(steps=50000, max_agents=1000)  # rwh

    # 2. Open the interactive widget viewer
    sim.interactive_viewer(
        title="Growing Monolayer",
        interval=5,   # 120,   # rwh: ms between auto-play frames
    )
