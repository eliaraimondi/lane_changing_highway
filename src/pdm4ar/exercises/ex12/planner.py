from mimetypes import init
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleModel
from dg_commons import SE2Transform
import numpy as np
import math
from dg_commons.sim.models.vehicle import VehicleState
from .structures import *
import matplotlib.pyplot as plt
import copy


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
    acc = (goal_speed**2 - current_speed**2) / 2 * trajectory_length

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


def compute_vehicle_states(path: Path, init_speed: float, goal_speed: float, radius: float, wheelbase: float) -> dict:
    """
    Compute the vehicle states from the path
    """
    # Compute the total length of the path
    trajectory_length = 0
    for segment in path:
        trajectory_length += segment.length

    # Compute the constant acceleration to reach the goal speed
    acc = (goal_speed**2 - init_speed**2) / 2 * trajectory_length

    # Convert the path to a list of states
    trajectory = extract_path_points(path)

    # Compute the time to reach the goal speed
    if acc == 0:
        total_time = trajectory_length / init_speed
    else:
        discriminant = init_speed**2 + 2 * acc * trajectory_length
        total_time = (-init_speed + math.sqrt(discriminant)) / acc

    # Compute the speed for each point in the trajectory
    speeds = np.linspace(init_speed, goal_speed, len(trajectory))

    # Compute the different time steps for each point in the trajectory
    time_steps = np.linspace(0, total_time, len(trajectory))

    # Create the dict where 0.1 are the keys and the values are the states
    vehicle_states = {}
    for j in range(int(np.floor(total_time * 10) + 1)):
        t = j / 10
        for i, time in enumerate(time_steps):
            if t == 0:
                new_x = trajectory[i].p[0]
                new_y = trajectory[i].p[1]
                new_psi = trajectory[i].theta
                new_v = speeds[i]
                new_delta = 0
                break
            if time == t:
                new_x = trajectory[i].p[0]
                new_y = trajectory[i].p[1]
                new_psi = trajectory[i].theta
                new_v = speeds[i]
                delta_psi = trajectory[i - 1].theta - trajectory[i].theta
                dpsi = delta_psi / (time_steps[i] - time_steps[i - 1])
                new_delta = math.atan((wheelbase * dpsi) / new_v)
                break
            if time < t <= time_steps[i + 1]:
                dt = t - time
                # Compute the distance along the curve axis
                delta_s = speeds[i] * dt + 0.5 * acc * dt**2

                if trajectory[i].p[0] <= path[2].start_config.p[0]:  # consider the two curves
                    # Decompose the distance along the curve axis into x and y components
                    alpha = delta_s / radius
                    chord_length = 2 * radius * math.sin(alpha / 2)

                    # Find the center of the turning circle
                    if trajectory[i].p[0] < path[1].start_config.p[0]:
                        center = path[0].center.p
                    else:
                        center = path[1].center.p

                    # Compute the new x and y
                    alpha_radius = math.atan2(center[1] - trajectory[i].p[1], center[0] - trajectory[i].p[0])
                    ang_int = (np.pi - alpha) / 2
                    my_ang = alpha_radius + ang_int
                    new_x = trajectory[i].p[0] + chord_length * math.cos(my_ang)
                    new_y = trajectory[i].p[1] + chord_length * math.sin(my_ang)

                    # Compute the new psi and v
                    old_delta_s = speeds[i] * (time_steps[i + 1] - time) + 0.5 * acc * (time_steps[i + 1] - time) ** 2
                    new_psi = trajectory[i].theta + (delta_s / old_delta_s) * (
                        mod_2_pi(trajectory[i + 1].theta) - mod_2_pi(trajectory[i].theta)
                    )
                    new_v = speeds[i] + acc * dt

                    # Compute the new delta
                    if np.abs(new_psi - trajectory[i].theta) > 5:
                        delta_psi = new_psi - trajectory[i].theta - 2 * np.pi
                    else:
                        delta_psi = new_psi - trajectory[i].theta
                    dpsi = delta_psi / dt
                    new_delta = math.atan((wheelbase * dpsi) / new_v)
                    break

                else:  # condider the straight line
                    # Compute the new x and y
                    new_x = trajectory[i].p[0] + delta_s * math.cos(trajectory[i].theta)
                    new_y = trajectory[i].p[1] + delta_s * math.sin(trajectory[i].theta)

                    # Compute the new psi and v
                    old_delta_s = speeds[i] * (time_steps[i + 1] - time) + 0.5 * acc * (time_steps[i + 1] - time) ** 2
                    new_psi = path[2].start_config.theta
                    new_v = speeds[i] + acc * dt
                    new_delta = 0
                    break

        vehicle_states[t] = VehicleState(x=new_x, y=new_y, psi=mod_2_pi(new_psi), vx=new_v, delta=new_delta)

    x_to_plot = [point.x for point in vehicle_states.values()]
    y_to_plot = [point.y for point in vehicle_states.values()]
    plt.figure()
    plt.plot(x_to_plot, y_to_plot)
    plt.savefig("vehicle_states.png")

    new_states = smooth_delta(vehicle_states)

    return new_states


def extract_path_points(path: Path) -> list[SE2Transform]:
    """Extracts a fixed number of SE2Transform points on a path"""
    pts_list = []
    num_points_per_segment = 20
    for idx, seg in enumerate(path):
        # if np.allclose(seg.length, 0):
        #     continue
        seg.start_config.theta = mod_2_pi(seg.start_config.theta)
        seg.end_config.theta = mod_2_pi(seg.end_config.theta)
        if seg.type is DubinsSegmentType.STRAIGHT:
            line_pts = interpolate_line_points(seg, num_points_per_segment)
            pts_list.extend(line_pts)
        else:  # Curve
            curve_pts = interpolate_curve_points(seg, num_points_per_segment)
            pts_list.extend(curve_pts)
    pts_list.append(path[-1].end_config)
    return pts_list


def interpolate_line_points(line: Line, number_of_points: float) -> list[SE2Transform]:
    start = line.start_config
    end = line.end_config
    start_to_end = end.p - start.p
    intervals = np.linspace(0, 1.0, number_of_points)
    return [SE2Transform(start.p + i * start_to_end, start.theta) for i in intervals]


def interpolate_curve_points(curve: Curve, number_of_points: int) -> list[SE2Transform]:
    pts_list = []
    angle = curve.arc_angle
    direction = curve.type
    angle = direction.value * angle
    split_angle = angle / number_of_points
    old_point = curve.start_config
    for _ in range(number_of_points):
        pts_list.append(old_point)
        point_next = get_next_point_on_curve(curve, point=old_point, delta_angle=split_angle)
        old_point = point_next
    return pts_list


def get_rot_matrix(alpha: float) -> np.ndarray:
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_matrix


def get_next_point_on_curve(curve: Curve, point: SE2Transform, delta_angle: float) -> SE2Transform:
    point_translated = point.p - curve.center.p
    rot_matrix = get_rot_matrix(delta_angle)
    next_point = SE2Transform((rot_matrix @ point_translated) + curve.center.p, point.theta + delta_angle)
    return next_point


"""def smooth_delta(trajectory: dict) -> dict:
    new_trajectory = copy.deepcopy(trajectory)
    values = list(trajectory.values())
    error = 0
    max_dif_delta = 0.05
    for i in range(len(values) - 1):
        if values[i + 1].delta - new_trajectory[i / 10].delta > max_dif_delta and values[i + 1].delta > 0:
            new_trajectory[(i + 1) / 10].delta = new_trajectory[i / 10].delta + max_dif_delta
            error += values[i + 1].delta - new_trajectory[(i + 1) / 10].delta
        elif values[i + 1].delta - new_trajectory[i / 10].delta < -max_dif_delta and values[i + 1].delta < 0:
            new_trajectory[(i + 1) / 10].delta = new_trajectory[i / 10].delta - max_dif_delta
            error += np.abs(values[i + 1].delta - new_trajectory[(i + 1) / 10].delta)
        else:
            additional_error = min(error, max_dif_delta)
            if values[i + 1].delta > 0:
                new_trajectory[(i + 1) / 10].delta = min(
                    values[i + 1].delta + additional_error, new_trajectory[i / 10].delta + max_dif_delta
                )
            else:
                new_trajectory[(i + 1) / 10].delta = max(
                    values[i + 1].delta - additional_error, new_trajectory[i / 10].delta - max_dif_delta
                )
            error -= np.abs(values[i + 1].delta - new_trajectory[(i + 1) / 10].delta)

    return new_trajectory"""


def smooth_delta(trajectory: dict) -> dict:
    new_trajectory = copy.deepcopy(trajectory)
    delta = trajectory[0.1].delta
    tf = round(((len(trajectory) - 1) / 10) / 3 * 2, 1)
    # compute sin and approximate for the number of element of trajectory
    for ts in range(0, int(tf * 10) + 1):
        ts = round(float(ts / 10), 1)
        new_delta = np.pi / 2 * delta * np.sin(2 * np.pi / tf * ts)
        new_trajectory[ts].delta = new_delta
    return new_trajectory
