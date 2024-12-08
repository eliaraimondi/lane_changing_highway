import random
from dataclasses import dataclass
from typing import Sequence
from dg_commons import SE2Transform

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleModel
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from .dubins import calculate_dubins_path, calculate_car_turning_radius, extract_path_points
from .collision_checking import collision_checking
from dg_commons import SE2Transform
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    sg: VehicleGeometry
    sp: VehicleParameters
    min_turning_radius: float
    start: SE2Transform
    goal: SE2Transform
    path: Sequence[SE2Transform]

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        self.min_turning_radius = calculate_car_turning_radius(
            self.sg.lr + self.sg.lf, self.sp.delta_max
        ).min_radius  # calculate the minimum turning radius of the car

        # Create a dictionary to store the speeds of other agents
        self.old_other_speeds = {}

        self.numb_of_steps = 61  # number of points for the trajectory (equal to number of points of our trajectory)

        lanelet_network = init_obs.dg_scenario.lanelet_network
        self.control_points = init_obs.goal.ref_lane.control_points
        starting_lanelet_id = lanelet_network.find_lanelet_by_position([self.control_points[1].q.p])[0][0]

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        # State at the current time
        self.current_state = sim_obs.players[self.name].state

        # COMPUTE SPEEDS AND ACCELERATIONS OF OTHER AGENTS
        # Save speeds of other agents to compuete their accelerations
        self.other_speeds = {
            agent_name: sim_obs.players[agent_name].state.vx
            for agent_name in sim_obs.players
            if agent_name != self.name
        }

        # If have old speeds, compute accelerations
        if self.old_other_speeds:
            self.other_accelerations = {
                agent_name: (self.other_speeds[agent_name] - self.old_other_speeds[agent_name]) / 0.1
                for agent_name in self.other_speeds
            }
        else:
            # If it is the first iteration, set the car radius and initialize the other accelerations
            self.my_radius = sim_obs.players[self.name].occupancy.length / 2
            self.other_accelerations = {}

        # Update old speeds
        self.old_other_speeds = self.other_speeds

        # COMPUTE POINTS OF TRAJECTORY OF OTHER AGENTS
        # Create a dictionary to store the trajectories of other agents
        other_trajectories = {}

        # For each agent, compute the trajectory
        for agent_name in sim_obs.players:
            agent = sim_obs.players[agent_name]

            # If we don't have accelleration of the agent, set it to 0
            if agent_name not in self.other_accelerations:
                self.other_accelerations[agent_name] = 0

            if agent_name != self.name:
                # Compute cumulative delta s for each agent every 0.1 seconds
                # Use the formula for the accelerated motion to compute the cumulative delta s
                cumulative_delta_s = [
                    agent.state.vx * 0.1 * step + 0.5 * self.other_accelerations[agent_name] * (0.1 * step) ** 2
                    for step in range(self.numb_of_steps)
                ]

                # Decompose the cumulative delta s into x and y components
                other_trajectories[agent_name] = [
                    (
                        cumulative_delta_s[step] * np.cos(agent.state.psi) + agent.state.x,
                        cumulative_delta_s[step] * np.sin(agent.state.psi) + agent.state.y,
                    )
                    for step in range(self.numb_of_steps)
                ]

        ############################################################################################################

        # COMMANDS
        # Generate random commands
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1
        commands = VehicleCommands(rnd_acc, rnd_ddelta)

        current_state = sim_obs.players[self.name].state
        initial_state = SE2Transform([current_state.x, current_state.y], current_state.psi)
        final_state = self.set_goal()

        path = calculate_dubins_path(initial_state, final_state, self.min_turning_radius)
        trajectory = extract_path_points(path)
        self.trajectory_to_plot = [[tr.p[0] for tr in trajectory], [tr.p[1] for tr in trajectory]]

        self._plot_print()

        ############################################################################################################
        # CHECK IF THE TRAJECTORY OF THE AGENT INTERSECTS WITH THE TRAJECTORIES OF OTHER AGENTS
        # 1. Trasform the trajectory in a list of tuples
        trajectory_points = [(tr.p[0], tr.p[1]) for tr in trajectory]  # need to use the points every 0.1 seconds
        # 2. Check if the trajectory intersects with the trajectories of other agents
        for agent_name in other_trajectories:
            agent_radius = sim_obs.players[agent_name].occupancy.length / 2
            collisions = collision_checking(
                trajectory_points, other_trajectories[agent_name], self.my_radius, agent_radius
            )

        return commands

    def set_goal(self) -> SE2Transform:
        """
        This method sets the goal of the agent.
        """
        width_line = 2 * self.control_points[-1].r
        s = 5
        m_y = self.control_points[-1].q.p[1] - self.control_points[-2].q.p[1]
        m_x = self.control_points[-1].q.p[0] - self.control_points[-2].q.p[0]
        scaler = max(abs(m_x), abs(m_y))
        m_x = m_x / scaler
        m_y = m_y / scaler
        theta = np.arctan2(m_y, m_x)
        x0 = self.current_state.x - width_line * np.sin(theta)
        y0 = self.current_state.y - width_line * np.cos(theta)
        x = x0 + s * m_x
        y = y0 + s * m_y
        goal = SE2Transform([x, y], theta)
        return goal

    def _plot_print(self):
        plt.figure()
        plt.plot(self.trajectory_to_plot[0], self.trajectory_to_plot[1])
        plt.axis("equal")
        plt.savefig("traiettoria.png")
