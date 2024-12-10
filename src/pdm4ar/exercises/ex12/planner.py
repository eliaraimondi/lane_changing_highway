from dg_commons.sim.models.vehicle import VehicleCommands, VehicleModel
from dg_commons import SE2Transform
import numpy as np
import math


def compute_commands(current_speed, goal_speed, trajectory, wheelbase: float, init_time: float) -> dict:
    """
    This method is called by the agent at each simulation step to compute the control commands for the vehicle.
    """
    # Initialize the dict of commands where the keys are the time steps and the values are the commands
    commands = {}

    trajectory_length = 0
    for i in range(len(trajectory) - 1):
        # Compute trajectory lenght
        x_curr, y_curr = trajectory[i].p
        x_next, y_next = trajectory[i + 1].p
        trajectory_length += np.sqrt((x_next - x_curr) ** 2 + (y_next - y_curr) ** 2)

    # Compute the constant acceleration to reach the goal speed
    acc = (goal_speed - current_speed) / 2 * trajectory_length

    # Compute total time to compleate the trajectory
    if acc == 0:
        total_time = trajectory_length / current_speed  # Moto uniforme
        dt = total_time / (len(trajectory) - 1)
    else:
        discriminant = current_speed**2 + 2 * acc * trajectory_length
        total_time = (-current_speed + math.sqrt(discriminant)) / acc
        dt = total_time / (len(trajectory) - 1)

    # Compute the speed for each point in the trajectory
    for i in range(len(trajectory) - 1):
        speeds = np.linspace(current_speed, goal_speed, len(trajectory))

    delta = [0.0]
    for i in range(len(trajectory) - 1):
        # Compute delta for each point in the trajectory
        delta_psi = trajectory[i + 1].theta - trajectory[i].theta
        dpsi = delta_psi / dt
        istantaneus_speed = (speeds[i] + speeds[i + 1]) / 2
        delta.append(math.atan((wheelbase * dpsi) / istantaneus_speed))

    ddelta = []
    for i in range(len(trajectory) - 1):
        # Compute ddelta for each point in the trajectory
        ddelta.append((delta[i + 1] - delta[i]) / dt)

    # Create the list of commands
    for i in range(len(trajectory) - 1):
        key = total_time / (len(trajectory) - 1) * i + init_time
        commands[key] = VehicleCommands(acc=acc, ddelta=ddelta[i])

    return commands
