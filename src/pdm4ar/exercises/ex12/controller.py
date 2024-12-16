from turtle import distance
import numpy as np
import math
import matplotlib.pyplot as plt
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
import casadi as ca


class Controller:
    def __init__(self, scenario, sp: VehicleParameters, sg: VehicleGeometry, dt, name, orientation, lane_width: float):
        self.dt = dt
        self.scenario = scenario
        self.sp = sp
        self.sg = sg
        self.name = name
        self.orientation = orientation
        self.orientations = []
        self.times = []
        self.cos_theta = np.cos(self.orientation)
        self.sin_theta = np.sin(self.orientation)
        self.lane_width = lane_width

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

    def maintain_lane(self, current_state: VehicleState, sim_obs, init: bool = False) -> VehicleCommands:
        """
        This method is called by the simulator to mantein the lane
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the actual commands to be executed
        """
        # Find the car in front of the agent
        front_car = self._find_front_car(current_state, sim_obs)
        max_speed = 25

        # Compute the actual acceleration
        if front_car is None:
            acc = max(min((max_speed - current_state.vx) / self.dt, self.sp.acc_limits[1]), self.sp.acc_limits[0])
        elif isinstance(front_car, VehicleCommands):
            return front_car
        else:
            # Compute the distance to cover
            distance_to_cover = (
                np.sqrt((front_car.state.x - current_state.x) ** 2 + (front_car.state.y - current_state.y) ** 2)
                - 2 * self.sg.length
            )
            max_speed = self._compute_max_speed(distance_to_cover, front_car.state.vx, current_state.vx)
            acc = max(min(self.sp.acc_limits[1], (max_speed - current_state.vx) / self.dt), self.sp.acc_limits[0])

        if not init:
            # Compute the actual ddelta
            delta_psi = self.orientation - current_state.psi
            dpsi = delta_psi / self.dt
            delta = math.atan((self.sg.wheelbase * dpsi) / current_state.vx)
            ddelta = min(self.sp.ddelta_max, (delta - current_state.delta) / self.dt)
        else:
            ddelta = 0

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

        return VehicleCommands(acc=acc, ddelta=ddelta)

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
        max_speed = 25
        max_dec = self.sp.acc_limits[0]

        # Compute the distance at the next state considering my current speed and the speed of the car in front
        distance = distance_to_cover + (speed_goal - current_speed) * self.dt

        # Compute the maximum speed considering the distance to cover
        if distance > 0:
            speed_at_next_state = max(
                min(max_speed, np.sqrt(speed_goal**2 - 2 * max_dec * distance)), self.sp.vx_limits[0]
            )
        else:
            speed_at_next_state = 0

        return speed_at_next_state

    def _find_front_car(self, current_state: VehicleState, sim_obs):
        """
        This function finds the car in front of the agent
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the car in front of the agent, None if there is no car in front
        """
        # Find the lanelet of the agent
        my_position = [np.array([current_state.x, current_state.y])]
        try:
            my_lanelet_ID = self.scenario.find_lanelet_by_position(my_position)[0][0]
        except IndexError:
            return VehicleCommands(acc=0, ddelta=0)

        # Find the predecessor and the successor of the lanelet
        my_lanelet_IDs = self.successor_and_predecessor(my_lanelet_ID)

        # Find the car in front of the agent
        for name, agent in sim_obs.players.items():
            if name != self.name:
                agent_position = [np.array([agent.state.x, agent.state.y])]

                # Try to find the lanelet of the agent, if the list is empty continue
                try:
                    agent_lanelet = self.scenario.find_lanelet_by_position(agent_position)[0][0]
                except IndexError:
                    continue
                if (agent_lanelet in my_lanelet_IDs) and (
                    (agent.state.x > current_state.x and np.cos(self.orientation) > 0)
                    or (agent.state.x < current_state.x and np.cos(self.orientation) < 0)
                ):
                    return agent

        # If there is no car in front of the agent, return None
        return None

    def successor_and_predecessor(self, my_lanelet_ID: int) -> tuple:
        """
        This function returns the successor and the predecessor of the goal lanelet
        :param my_lanelet_ID: the ID of the lanelet of the agent
        :return: the successor and the predecessor of the goal lanelet
        """
        my_lanelet = self.scenario.find_lanelet_by_id(my_lanelet_ID)
        try:
            lane_suc_ID = my_lanelet.successor[0]
        except:
            lane_suc_ID = None
        try:
            lane_pre_ID = my_lanelet.predecessor[0]
        except:
            lane_pre_ID = None

        return (lane_suc_ID, lane_pre_ID, my_lanelet_ID)

    def control_in_wall_cars(
        self, current_state: VehicleState, sim_obs, goal_IDs, goal_lane_is_right: bool
    ) -> VehicleCommands:
        """
        This method is called by the simulator to mantein the lane in the wall cars condition
        :param current_state: the current state of the agent at the current time step
        :param sim_obs: the current observations of the simulator
        :return: the actual commands to be executed
        """
        project_segment = self.lane_width

        agents_in_goal = []
        agents = sim_obs.players
        for agent_name, agent in agents.items():
            position = [np.array([agent.state.x, agent.state.y])]
            lanelet = self.scenario.find_lanelet_by_position(position)[0][0]
            if lanelet in goal_IDs:
                agents_in_goal.append(agent_name)

        # Project my position in the goal lane
        angle = self.orientation
        if goal_lane_is_right:
            angle -= np.pi / 2
        else:
            angle += np.pi / 2

        my_x_in_goal = current_state.x + project_segment * np.cos(angle)
        my_y_in_goal = current_state.y + project_segment * np.sin(angle)

        # Find the car in front of the agent
        distances = {}
        for agent_name in agents_in_goal:
            if (agent.state.x > my_x_in_goal and np.cos(self.orientation) > 0) or (
                agent.state.x < my_x_in_goal and np.cos(self.orientation) < 0
            ):
                distances[agent_name] = np.sqrt(
                    (agent.state.x - my_x_in_goal) ** 2 + (agent.state.y - my_y_in_goal) ** 2
                )
        self.closest_car = min(distances, key=distances.get)

        # Compute the acceleration
        other_car_speed = agents[self.closest_car].state.vx
        other_car_next_state = other_car_speed * self.dt + agents[self.closest_car].state.x
        distance_to_cover = other_car_next_state - current_state.x
        speed_next_state = self._compute_max_speed(distance_to_cover, other_car_speed, current_state.vx)
        acc = max(min(self.sp.acc_limits[1], (speed_next_state - current_state.vx) / self.dt), self.sp.acc_limits[0])

    def cruise_control(self, current_state: VehicleState, init_time: float) -> dict[float, VehicleCommands]:
        # Parameters
        N = 10  # Prediction horizon
        Q_y = 100.0  # Lane deviation cost weight
        x_proj = -self.sin_theta
        y_proj = self.cos_theta

        my_position = [np.array([current_state.x, current_state.y])]
        try:
            my_lanelet_ID = self.scenario.find_lanelet_by_position(my_position)[0][0]
            my_lanelet = self.scenario.find_lanelet_by_id(my_lanelet_ID)
            x_control_p = my_lanelet.center_vertices[0][0]  # take first control point
            y_control_p = my_lanelet.center_vertices[0][1]  # ""
        except IndexError:
            return {round((i * 0.1) + init_time, 1): VehicleCommands(acc=0, ddelta=0) for i in range(N)}

        # Define the states
        psi = ca.MX.sym("psi")
        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        vx = ca.MX.sym("vx")
        delta = ca.MX.sym("delta")
        states = ca.vertcat(psi, x, y, vx, delta)
        n_states = states.size1()

        # Define the controls
        a = ca.MX.sym("a")
        ddelta = ca.MX.sym("ddelta")
        controls = ca.vertcat(a, ddelta)
        n_controls = controls.size1()

        # Define discretized dynamics
        dpsi = vx * ca.tan(delta) / self.sg.wheelbase
        # vy = dpsi * self.sg.lr
        # costh = ca.cos(psi)
        # sinth = ca.sin(psi)
        xdot = vx * ca.cos(psi) - dpsi * self.sg.lr * ca.sin(psi)
        ydot = vx * ca.sin(psi) + dpsi * self.sg.lr * ca.cos(psi)

        rhs = ca.vertcat(dpsi, xdot, ydot, a, ddelta)

        f = ca.Function("f", [states, controls], [rhs])

        # Optimization problem
        X = ca.MX.sym("X", n_states, N + 1)
        U = ca.MX.sym("U", n_controls, N)
        lane_error = ca.MX.sym("lane_error", 1, N)  # Define lane_position_error: NOTE: starts for X[:, 1]

        P = ca.MX.sym("P", n_states + 1)  # Initial state + psi_desired (column vector)

        obj = 0
        g = []  # constraints

        for k in range(N):
            # Cost function
            obj += Q_y * lane_error[0, k] ** 2  # + Q_psi * psi_error ** 2
            # obj += R_a * U[0, k] ** 2 + R_delta * X[4, k + 1] ** 2 + R_ddelta * U[1, k] ** 2
            # Dynamics constraint
            x_next = X[:, k] + self.dt * f(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)  # NOTE: each constr is vectorized --> 10 constraints for 50 actual

        # Lane deviation constraint
        for k in range(N):
            g.append(x_proj * (X[1, k + 1] - x_control_p) + y_proj * (X[2, k + 1] - y_control_p) - lane_error[k])

        # Initial state
        g.append(X[:, 0] - P[:n_states])

        # Final Orientation, delta
        g.append(X[0, N] - P[n_states])
        g.append(X[4, N])

        # Flatten constraints
        g = ca.vertcat(*g)

        # Bounds for states and controls
        lbx = []
        ubx = []

        for _ in range(N + 1):  # State bounds
            lbx += [-ca.inf, -ca.inf, -ca.inf, self.sp.vx_limits[0], -self.sp.delta_max]  # [psi, x, y, v, delta]
            ubx += [ca.inf, ca.inf, ca.inf, self.sp.vx_limits[1], self.sp.delta_max]  # [psi, x, y, v, delta]

        for _ in range(N):  # Control bounds
            lbx += [self.sp.acc_limits[0], -self.sp.ddelta_max]  # [a, dot_delta]
            ubx += [self.sp.acc_limits[1], self.sp.ddelta_max]  # [a, dot_delta]

        for _ in range(N):  # Lane deviation bounds
            lbx += [-ca.inf]
            ubx += [ca.inf]

        # Pack variables for solver
        opt_variables = ca.vertcat(
            ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), ca.reshape(lane_error, -1, 1)
        )  # column vectors: all states in each instant --> NOTE: ca.reshape follows FORTRAN order!

        # Nonlinear programming problem
        nlp = {"x": opt_variables, "f": obj, "g": g, "p": P}
        solver = ca.nlpsol("solver", "ipopt", nlp)

        ## SOLVE
        # Initial guess
        x0 = current_state.as_ndarray()  # Initial state [x, y, psi, v, delta]
        psi0 = x0[2]
        x0[2] = x0[1]
        x0[1] = x0[0]
        x0[0] = psi0  # Initial state [psi, x, y, v, delta]
        u0 = np.zeros((n_controls, N))  # Initial guess for controls
        X0 = np.tile(x0, (N + 1, 1)).T  # Initial guess for states
        lane_error0 = np.array([x_proj * (x0[0] - x_control_p) + y_proj * (x0[1] - y_control_p)])
        lane_error0 = np.tile(lane_error0, (N, 1)).T
        # Fill parameter vector
        p = np.hstack((x0, self.orientation))

        sol = solver(
            x0=ca.vertcat(ca.reshape(X0, -1, 1), ca.reshape(u0, -1, 1), ca.reshape(lane_error0, -1, 1)),
            lbx=lbx,
            ubx=ubx,
            lbg=0,
            ubg=0,
            p=p,
        )

        # Extract optimal solution
        opt_X = np.array(sol["x"][: n_states * (N + 1)]).reshape(N + 1, n_states).T
        opt_U = np.array(sol["x"][n_states * (N + 1) : n_states * (N + 1) + n_controls * N]).reshape(N, n_controls).T
        opt_lane_error = np.array(sol["x"][n_states * (N + 1) + n_controls * N :]).reshape(1, N)
        dict = {round((i * 0.1) + init_time, 1): VehicleCommands(acc=opt_U[0, i], ddelta=opt_U[1, i]) for i in range(N)}

        return dict
