from pickle import TRUE
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
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleModel, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

from pdm4ar.exercises.ex05.structures import mod_2_pi
from .dubins import calculate_dubins_path, calculate_car_turning_radius
from .collision_checking import CollisionChecker
from dg_commons import SE2Transform
import matplotlib.pyplot as plt
import numpy as np
import math


@dataclass(frozen=True)
class Pdm4arAgentParams:
    a = 0  # number of points for the trajectory (equal to number of points of our trajectory)


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
        self.cars_on_same_lanelet = {}  # Name: PlayersObservations: state, occupancy
        self.cars_on_goal_lanelet = {}  # ""
        # Create a dictionary to store the trajectories of other agents
        self.other_trajectories = {}
        self.portion_of_trajectory = 3
        self.collision_checker = CollisionChecker(self.portion_of_trajectory)
        self.cars_already_seen = []
        self.trajectory_started = False
        self.params = Pdm4arAgentParams()
        self.old_other_speeds = {}

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        self.min_radius = calculate_car_turning_radius(
            self.sg.lr + self.sg.lf, self.sp.delta_max
        ).min_radius  # calculate the minimum turning radius of the car

        # Create a dictionary to store the speeds of other agents
        self.scenario: LaneletNetwork = init_obs.dg_scenario.lanelet_network
        self.control_points = init_obs.goal.ref_lane.control_points
        self.goal_ID = self.scenario.find_lanelet_by_position([self.control_points[1].q.p])[0][0]
        self.orientation = self.goal.ref_lane.control_points[0].q.theta

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        current_state: VehicleState = sim_obs.players[self.name].state

        # If the trajectory is not started, compute the trajectory
        if not self.trajectory_started:
            my_position = [np.array([current_state.x, current_state.y])]
            my_lanelet = self.scenario.find_lanelet_by_position(my_position)[0][0]

            ### Update neighborhood ###
            self.cars_on_same_lanelet.clear()
            self.cars_on_goal_lanelet.clear()
            front_on_same, front_on_goal = None, None  # PlayersObservations: state, occupancy
            rear_on_same, rear_on_goal = None, None

            for cars in sim_obs.players:
                if cars == self.name:
                    continue
                car = sim_obs.players[cars]
                car_position = [np.array([car.state.x, car.state.y])]
                car_lanelet = self.scenario.find_lanelet_by_position(car_position)[0][0]
                if car_lanelet == my_lanelet:  # NOTE: USELESS IF MY_LANELET IS THE GOAL LANELET
                    self.cars_on_same_lanelet[cars] = car
                    if car.state.x > current_state.x and (
                        self.orientation < np.pi / 2 or self.orientation > 3 * np.pi / 2
                    ):  # from sx to dx (NOTE: NO vertical lane because check on x-coordinates)
                        front_on_same = (
                            car if front_on_same is None or car.state.x < front_on_same.state.x else front_on_same
                        )
                    elif car.state.x < current_state.x and (
                        self.orientation < np.pi / 2 or self.orientation > 3 * np.pi / 2
                    ):
                        rear_on_same = (
                            car if rear_on_same is None or car.state.x > rear_on_same.state.x else rear_on_same
                        )
                    continue
                if car_lanelet == self.goal_ID:
                    self.cars_on_goal_lanelet[cars] = car
                    if car.state.x > current_state.x and (
                        self.orientation < np.pi / 2 or self.orientation > 3 * np.pi / 2
                    ):
                        front_on_goal = (
                            car if front_on_goal is None or car.state.x < front_on_goal.state.x else front_on_goal
                        )
                    elif car.state.x < current_state.x and (
                        self.orientation < np.pi / 2 or self.orientation > 3 * np.pi / 2
                    ):
                        rear_on_goal = (
                            car if rear_on_goal is None or car.state.x > rear_on_goal.state.x else rear_on_goal
                        )
                    elif car.state.x < current_state.x and (
                        self.orientation > np.pi / 2 or self.orientation < 3 * np.pi / 2
                    ):
                        front_on_goal = (
                            car if front_on_goal is None or car.state.x > front_on_goal.state.x else front_on_goal
                        )
                    elif car.state.x > current_state.x and (
                        self.orientation > np.pi / 2 or self.orientation < 3 * np.pi / 2
                    ):
                        rear_on_goal = (
                            car if rear_on_goal is None or car.state.x < rear_on_goal.state.x else rear_on_goal
                        )
                    continue

            ############################################################################################################
            # TRAJECTORY
            # Initial state for the dubins
            current_state = sim_obs.players[self.name].state
            # Final speed of the car
            # Final speed of the car
            end_speed = current_state.vx
            if front_on_goal is not None:
                end_speed = front_on_goal.state.vx

            # Check if the goal lane is on the right side of the car
            goal_lane_is_right = self.point_is_right(
                current_state.x,
                current_state.y,
                current_state.psi,
                self.control_points[1].q.p[0],
                self.control_points[1].q.p[1],
            )

            lane_width = 2 * self.control_points[1].r
            radius = self.min_radius * lane_width * 5

            self.trajectory = calculate_dubins_path(
                current_state,
                end_speed,
                radius,
                goal_lane_is_right,
                lane_width=lane_width,
                wheelbase=self.sg.wheelbase,
            )

            ############################################################################################################
            # CHECK IF THE TRAJECTORY OF THE AGENT INTERSECTS WITH THE TRAJECTORIES OF OTHER AGENTS
            # 0. Compute the trajectories of other agents
            self.compute_other_trajectories(sim_obs)

            # Add the new_geometries to the collision checker
            for agent_name in sim_obs.players:
                if agent_name not in self.cars_already_seen:
                    if agent_name == self.name:
                        self.collision_checker.add_other_geometry(agent_name, self.sg)
                    else:
                        self.collision_checker.add_other_geometry(agent_name, sim_obs.players[agent_name].occupancy)
                    self.cars_already_seen.append(agent_name)

            # 1. Trasform the trajectory in a list of tuples
            states = list(self.trajectory.values())
            trajectory_points = [
                SE2Transform([tr.x, tr.y], tr.psi) for tr in states
            ]  # need to use the points every 0.1 seconds

            # 2. Check if the trajectory intersects with the trajectories of other agents
            agents_collisions = self.collision_checker.collision_checking(
                trajectory_points,
                my_name=self.name,
                other_positions_dict=self.other_trajectories,
            )

            if all(not lst for lst in agents_collisions.values()):
                self.trajectory_started = True
            else:
                commands = VehicleCommands(acc=0, ddelta=0)

        # If the trajectory is started compute the commands
        ind = round(float(sim_obs.time) + 0.1, 1)
        if self.trajectory_started and list(self.trajectory.keys())[-1] > float(sim_obs.time):
            commands = self.compute_actual_commands(current_state, self.trajectory[ind])

        if self.trajectory_started and list(self.trajectory.keys())[-1] <= float(sim_obs.time):
            commands = VehicleCommands(acc=0, ddelta=0)

        return commands

    def point_is_right(self, current_x, current_y, current_psi, goal_x, goal_y):
        """
        This function checks if the goal lane is on the right side of the car
        :param current_x: x coordinate of the current position
        :param current_y: y coordinate of the current position
        :param current_psi: current orientation of the car
        :param goal_x: x coordinate of the goal position
        :param goal_y: y coordinate of the goal position
        :return: True if the goal is on the right side of the car, False otherwise
        """
        # Calculate the heading vector
        heading_x = math.cos(current_psi)
        heading_y = math.sin(current_psi)

        # Vector from current position to the goal
        dx = goal_x - current_x
        dy = goal_y - current_y

        # Compute the cross product
        cross = heading_x * dy - heading_y * dx

        # Return True if the goal is to the right
        return cross < 0

    def compute_actual_commands(self, current_state, desired_state) -> VehicleCommands:
        """
        This method is called by the simulator to compute the actual commands to be executed
        :param current_state: the current state of the agent at the current time step
        :param desired_state: the desired state of the agent at the next time step
        :return: the actual commands to be executed
        """
        # Compute the actual acceleration
        dt = 0.1
        acc = (desired_state.vx - current_state.vx) / dt

        # Compute the actual ddelta
        ddelta = (desired_state.delta - current_state.delta) / dt

        return VehicleCommands(acc=acc, ddelta=ddelta)

    def compute_other_trajectories(self, sim_obs):
        """
        This method computes the trajectories of other agents
        :param sim_obs: the current observations of the simulator
        :return: a dictionary containing the trajectories of other agents
        """
        # COMPUTE SPEEDS AND ACCELERATIONS OF OTHER AGENTS
        # Save speeds of other agents to compuete their accelerations
        numb_of_steps = len(self.trajectory) // self.portion_of_trajectory  # da tunnare

        """# Compute the speeds of other agents
        other_speeds = {
            agent_name: sim_obs.players[agent_name].state.vx
            for agent_name in sim_obs.players
            if agent_name != self.name
        }

        # If have old speeds, compute accelerations
        if self.old_other_speeds:
            self.other_accelerations = {
                agent_name: (other_speeds[agent_name] - self.old_other_speeds[agent_name]) / 0.1
                for agent_name in self.other_speeds
            }
        else:
            # If it is the first iteration, set the car radius and initialize the other accelerations
            self.my_radius = sim_obs.players[self.name].occupancy.length / 2
            self.other_accelerations = {}

        # Update old speeds
        self.old_other_speeds = self.other_speeds"""

        other_accelerations = {}
        # COMPUTE POINTS OF TRAJECTORY OF OTHER AGENTS
        # For each agent, compute the trajectory
        for agent_name in sim_obs.players:
            agent = sim_obs.players[agent_name]

            # If we don't have accelleration of the agent, set it to 0
            # if agent_name not in self.other_accelerations:
            other_accelerations[agent_name] = 0

            if agent_name != self.name:
                # Compute cumulative delta s for each agent every 0.1 seconds
                # Use the formula for the accelerated motion to compute the cumulative delta s
                cumulative_delta_s = [
                    agent.state.vx * 0.1 * step + 0.5 * other_accelerations[agent_name] * (0.1 * step) ** 2
                    for step in range(numb_of_steps)
                ]

                # Decompose the cumulative delta s into x and y components
                self.other_trajectories[agent_name] = [
                    SE2Transform(
                        [
                            cumulative_delta_s[step] * np.cos(agent.state.psi) + agent.state.x,
                            cumulative_delta_s[step] * np.sin(agent.state.psi) + agent.state.y,
                        ],
                        agent.state.psi,
                    )
                    for step in range(numb_of_steps)
                ]
