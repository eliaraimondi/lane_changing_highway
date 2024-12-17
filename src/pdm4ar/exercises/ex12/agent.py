from dataclasses import dataclass
from typing import Sequence
from cycler import V
from dg_commons import SE2Transform

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

from .dubins import DubinsPath
from .collision_checking import CollisionChecker
from .controller import Controller
from dg_commons import SE2Transform
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
    sg: VehicleGeometry
    sp: VehicleParameters
    min_turning_radius: float
    start: SE2Transform
    goal: SE2Transform
    path: Sequence[SE2Transform]

    def __init__(self):
        # Create a dictionary to store the trajectories of other agents
        self.other_trajectories = {}
        self.trajectory = {}
        self.num_acc_steps = 4
        self.portion_of_trajectory = 1
        self.cars_already_seen = []
        self.trajectory_started = False
        self.dt = 0.1
        self.radius_coeff = (
            2 / 3
        )  # coeff of confidence for the radius, bigger values are more conservative, the smallest one is 1/2
        self.wall_cars = False
        self.closest_car_name = None

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal  # type: ignore
        self.sg = init_obs.model_geometry  # type: ignore
        self.sp = init_obs.model_params  # type: ignore

        # Create a dictionary to store the speeds of other agents
        self.scenario: LaneletNetwork = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.control_points = init_obs.goal.ref_lane.control_points  # type: ignore
        point1 = self.control_points[0].q.p
        point2 = self.control_points[1].q.p
        self.orientation = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])  # type: ignore
        self.lane_width = 2 * self.control_points[1].r
        self.collision_checker = CollisionChecker(self.portion_of_trajectory, self.name)
        self.controller = Controller(
            self.scenario,
            self.sp,
            self.sg,
            dt=self.dt,
            name=self.name,
            lane_width=self.lane_width,
        )
        self.controller.set_orientation(self.orientation)
        self.goal_ID = self.scenario.find_lanelet_by_position([self.control_points[1].q.p])[0][0]
        self.goal_IDs = self.controller.successor_and_predecessor(self.goal_ID)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        current_state: VehicleState = sim_obs.players[self.name].state  # type: ignore

        try:
            self.my_ID = self.scenario.find_lanelet_by_position([np.array([current_state.x, current_state.y])])[0][0]
            self.my_control_points = self.scenario.find_lanelet_by_id(self.my_ID).center_vertices
        except IndexError:
            self.my_control_points = self.control_points

        # Check if the goal lane is on the right side of the car
        if float(sim_obs.time) == 0.0:
            self.goal_lane_is_right = self.point_is_right(
                current_state.x,
                current_state.y,
                current_state.psi,
                self.control_points[1].q.p[0],
                self.control_points[1].q.p[1],
            )
            self.dubins_planner = DubinsPath(
                self.sg.wheelbase,
                self.sp.acc_limits[1],
                self.goal_lane_is_right,
                self.lane_width,
                self.sp.delta_max,
                self.radius_coeff,
            )

            if self._wall_cars(sim_obs=sim_obs):
                self.wall_cars = True

        if self.wall_cars and not self.trajectory_started:
            commands = self.control_wall_cars(current_state, sim_obs)

        # If the trajectory is not started and the time is a mult of 0.3, compute the trajectory
        if not self.wall_cars and not self.trajectory_started and float(sim_obs.time) % 0.3 == 0:

            ############################################################################################################
            # TRAJECTORY
            trajectory_not_scaled = self.dubins_planner.calculate_dubins_path(current_state, current_state.vx)

            # Add to the trajectory the actuale time
            self.trajectory = {
                round(key + float(sim_obs.time), 1): value for key, value in trajectory_not_scaled.items()
            }

            ############################################################################################################
            # CHECK IF THE TRAJECTORY OF THE AGENT INTERSECTS WITH THE TRAJECTORIES OF OTHER AGENTS
            # 0. Compute the trajectories of other agents
            self._compute_other_trajectories(sim_obs)

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
                other_positions_dict=self.other_trajectories,
                orientation=self.orientation,
            )

            if all(not lst for lst in agents_collisions.values()):
                self.trajectory_started = True
            else:
                commands = self.controller.maintain_lane(current_state, sim_obs, self.my_control_points)

        elif not self.trajectory_started and not self.wall_cars:
            commands = self.controller.maintain_lane(current_state, sim_obs, self.my_control_points)

        # If the trajectory is started compute the commands
        ind = round(float(sim_obs.time) + 0.1, 1)
        if self.trajectory_started and list(self.trajectory.keys())[-1] > float(sim_obs.time):
            commands = self.controller.compute_actual_commands(current_state, self.trajectory[ind])
        elif self.trajectory_started and list(self.trajectory.keys())[-1] <= float(sim_obs.time):
            commands = self.controller.maintain_lane(current_state, sim_obs, self.my_control_points)

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

    def _compute_other_trajectories(self, sim_obs):
        """
        This method computes the trajectories of other agents
        :param sim_obs: the current observations of the simulator
        :return: a dictionary containing the trajectories of other agents
        """
        # COMPUTE SPEEDS AND ACCELERATIONS OF OTHER AGENTS
        # Save speeds of other agents to compuete their accelerations
        numb_of_steps = int(len(self.trajectory) * self.portion_of_trajectory)  # da tunnare

        # COMPUTE POINTS OF TRAJECTORY OF OTHER AGENTS
        # For each agent, compute the trajectory
        for agent_name in sim_obs.players:
            agent = sim_obs.players[agent_name]

            if agent_name != self.name:
                # Compute cumulative delta s for each agent every 0.1 seconds
                # Use the formula for the accelerated motion to compute the cumulative delta s
                cumulative_delta_s = [agent.state.vx * 0.1 * step for step in range(numb_of_steps)]

                # Decompose the cumulative delta s into x and y components
                self.other_trajectories[agent_name] = [
                    SE2Transform(
                        [
                            cumulative_delta_s[step] * np.cos(self.orientation) + agent.state.x,
                            cumulative_delta_s[step] * np.sin(self.orientation) + agent.state.y,
                        ],
                        agent.state.psi,
                    )
                    for step in range(numb_of_steps)
                ]

    def _wall_cars(self, sim_obs) -> bool:
        """
        This function checks if there is a lot of cars in the goal lanelet
        :param sim_obs: the current observations of the simulator
        :return: True if there are a lot of cars in the goal lanelet, False otherwise
        """
        car_positions = []
        agents_in_goal = []
        agents = sim_obs.players
        for agent_name, agent in agents.items():
            position = [np.array([agent.state.x, agent.state.y])]
            try:
                lanelet = self.scenario.find_lanelet_by_position(position)[0][0]
            except IndexError:
                continue
            if lanelet in self.goal_IDs:
                agents_in_goal.append(agent_name)
                car_positions.append((agent.state.x, agent.state.y))
        distances_between_cars = []
        for i in range(len(car_positions) - 1):
            distances_between_cars.append(
                np.linalg.norm(
                    np.array([car_positions[i][0], car_positions[i][1]])
                    - np.array([car_positions[i + 1][0], car_positions[i + 1][1]])
                )
            )

        if len(agents_in_goal) > 5 or np.mean(distances_between_cars) < 2 * self.sg.length:
            return True
        else:
            return False

    def control_wall_cars(self, current_state: VehicleState, sim_obs):
        if self.closest_car_name is not None:
            if self.front_car_name is not None:
                distance = np.linalg.norm(
                    np.array(
                        [sim_obs.players[self.front_car_name].state.x, sim_obs.players[self.front_car_name].state.y]
                    )
                    - np.array(
                        [sim_obs.players[self.closest_car_name].state.x, sim_obs.players[self.closest_car_name].state.y]
                    )
                )
            else:
                distance = np.inf
            if distance == np.inf or (
                distance >= self.sg.length
                and round(current_state.vx - sim_obs.players[self.closest_car_name].state.vx, 1) == 0
                and round(self.distance_to_cover, 1) == 0
            ):
                trajectory_not_scaled = self.dubins_planner.calculate_dubins_path(current_state, 2 * current_state.vx)

                # Add to the trajectory the actuale time
                self.trajectory = {
                    round(key + float(sim_obs.time), 1): value for key, value in trajectory_not_scaled.items()
                }
                self.trajectory_started = True
                return VehicleCommands(acc=0, ddelta=0)

        commands, self.closest_car_name, self.front_car_name, self.distance_to_cover = (
            self.controller.maintain_in_wall_cars(
                current_state, sim_obs, self.goal_IDs, self.goal_lane_is_right, self.my_control_points
            )
        )
        return commands
