from dg_commons.sim.models.vehicle import VehicleCommands
import numpy as np
import math
import matplotlib.pyplot as plt
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters


class Controller:
    def __init__(self, scenario, sp: VehicleParameters, sg: VehicleGeometry, dt, name, orientation):
        self.dt = dt
        self.scenario = scenario
        self.sp = sp
        self.sg = sg
        self.name = name
        self.orientation = orientation
        self.orientations = []
        self.times = []

    def compute_actual_commands(self, current_state, desired_state) -> VehicleCommands:
        """
        This method is called by the simulator to compute the actual commands to be executed
        :param current_state: the current state of the agent at the current time step
        :param desired_state: the desired state of the agent at the next time step
        :return: the actual commands to be executed
        """
        # Compute the actual acceleration
        acc = (desired_state.vx - current_state.vx) / self.dt

        # Compute the actual ddelta
        ddelta = (desired_state.delta - current_state.delta) / self.dt

        return VehicleCommands(acc=acc, ddelta=ddelta)

    def maintain_lane(self, current_state, sim_obs):
        """
        This method is called by the simulator to mantein the lane
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the actual commands to be executed
        """
        my_position = [np.array([current_state.x, current_state.y])]
        try:
            my_lanelet = self.scenario.find_lanelet_by_position(my_position)[0][0]
        except:
            return VehicleCommands(acc=0, ddelta=0)
        front_car = None

        # Find the car in front of the agent
        for name, agent in sim_obs.players.items():
            if name != self.name:
                agent_position = [np.array([agent.state.x, agent.state.y])]

                # Try to find the lanelet of the agent, if the list is empty continue
                try:
                    agent_lanelet = self.scenario.find_lanelet_by_position(agent_position)[0][0]
                except:
                    continue
                if agent_lanelet == my_lanelet and (
                    (agent.state.x > current_state.x and np.cos(self.orientation) > 0)
                    or (agent.state.x < current_state.x and np.cos(self.orientation) < 0)
                ):
                    front_car = agent
                    break

        # If there isn't a car in front of the agent set all the commands to 0

        # Compute the actual acceleration
        if front_car is None:
            acc = self.sp.acc_limits[1]
        else:
            # Compute the distance to cover
            distance_to_cover = (
                np.sqrt((front_car.state.x - current_state.x) ** 2 + (front_car.state.y - current_state.y) ** 2)
                - 2 * self.sg.length
            )
            max_speed = self._compute_max_speed(distance_to_cover, front_car.state.vx, current_state.vx)

            acc = max(min(self.sp.acc_limits[1], (max_speed - current_state.vx) / self.dt), self.sp.acc_limits[0])

        # Compute the actual ddelta
        delta_psi = self.orientation - current_state.psi
        dpsi = delta_psi / self.dt
        delta = math.atan((self.sg.wheelbase * dpsi) / current_state.vx)
        ddelta1 = min(self.sp.ddelta_max, (delta - current_state.delta) / self.dt)

        """# Compute the orientation at the next step
        real_delta = ddelta1 * self.dt + current_state.delta
        real_dpsi = current_state.vx * math.tan(real_delta) / self.sg.wheelbase
        next_psi = real_dpsi * self.dt + current_state.psi
        next_delta_psi = self.orientation - next_psi
        next_dpsi = next_delta_psi / self.dt
        next_delta = math.atan((self.sg.wheelbase * next_dpsi) / (current_state.vx + acc * self.dt))
        ddelta2 = min(self.sp.ddelta_max, (delta - real_delta) / self.dt)"""

        # PID for the delta
        """error = delta - current_state.delta
        self.integral_pid += error * self.dt
        derivative = (error - self.error_previous) / self.dt
        ddelta = self.Kp * error  # + self.Ki * self.integral_pid + self.Kd * derivative"""

        self.orientations.append(current_state.psi)
        self.times.append(float(sim_obs.time))
        plt.figure()
        plt.plot(self.times, self.orientations)
        plt.savefig("orientation.png")

        return VehicleCommands(acc=acc, ddelta=ddelta1)

    def _compute_max_speed(self, distance_to_cover: float, speed_goal: float, current_speed: float) -> float:
        """
        This method computes the maximum speed of the agent considering the distance from the car in front
        of the agent
        :param distance_to_cover: the distance to cover to reach the car in front of the agent
        :param speed_goal: the current_speed of the agent
        :param current_speed: my current speed
        :return: the maximum speed of the agent
        """
        # Consider the maximum speed and dec of the agent
        max_speed = self.sp.vx_limits[1]
        max_dec = self.sp.acc_limits[0]

        # Compute the distance at the next state considering my current speed and the speed of the car in front
        distance = distance_to_cover + (speed_goal - current_speed) * self.dt

        # Compute the maximum speed considering the distance to cover
        if distance > 0:
            speed_at_next_state = min(max_speed, np.sqrt(speed_goal**2 - 2 * max_dec * distance))
        else:
            speed_at_next_state = 0

        return speed_at_next_state
