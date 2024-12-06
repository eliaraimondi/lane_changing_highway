import random
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleModel, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from .algo import calculate_dubins_path


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

        lanelet_network = init_obs.dg_scenario.lanelet_network
        control_points = init_obs.goal.ref_lane.control_points
        starting_lanelet_id = lanelet_network.find_lanelet_by_position([control_points[1].q.p])[0][0]

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        # todo implement here some better planning
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        current_car = VehicleModel.default_car(sim_obs.players[self.name].state)

        X_k1 = sim_obs.players[self.name].state + 0.1 * current_car.dynamics(
            x0=sim_obs.players[self.name].state,
            u=VehicleCommands.from_array(np.array([rnd_acc, rnd_ddelta])),
        )

        print(f"Current state: {sim_obs.players[self.name].state}")
        print(f"Next state: {X_k1}")
        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)
