from re import L
from typing import final
from dg_commons import SE2Transform
import math

import numpy as np
from traitlets import Bool

from .structures import *

from dg_commons.sim.models.vehicle import VehicleState
from .planner import compute_vehicle_states


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    min_radius = wheel_base / math.tan(max_steering_angle)

    return DubinsParam(min_radius)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    theta = current_config.theta
    x, y = current_config.p
    alpha = theta - math.pi / 2
    right_circle = Curve.create_circle(
        SE2Transform([x + radius * math.cos(alpha), y + radius * math.sin(alpha)], 0),
        current_config,
        radius,
        curve_type=DubinsSegmentType.RIGHT,
    )
    left_circle = Curve.create_circle(
        SE2Transform([x - radius * math.cos(alpha), y - radius * math.sin(alpha)], 0),
        current_config,
        radius,
        curve_type=DubinsSegmentType.LEFT,
    )

    return TurningCircle(left=left_circle, right=right_circle)


def calculate_dubins_path(
    init_config: VehicleState,
    end_speed: float,
    radius: float,
    goal_lane_is_right: bool,
    lane_width: float,
    wheelbase: float,
) -> dict:
    """
    Calculate the Dubins path with only 2 curves depending on the goal lane
    params:
    start_config: the start configuration of the car (x,y,theta)
    radius: the turning radius of the car
    goal_lane_is_right: a boolean indicating if the goal lane is on the right side of the car
    lane_width: the width of the lane
    """
    # Convert the start configuration to SE2Transform
    start_config = SE2Transform([init_config.x, init_config.y], init_config.psi)
    start_speed = init_config.vx

    # Compute the end configuration
    half_point_distance = np.sqrt(radius**2 - (radius - lane_width / 2) ** 2)
    end_config_x = start_config.p[0] + 2 * (
        half_point_distance * np.cos(start_config.theta) + lane_width / 2 * np.cos(start_config.theta + np.pi / 2)
    )

    if not goal_lane_is_right:
        # Compute path from the current lane to the right lane
        end_config_y = start_config.p[1] + 2 * (
            half_point_distance * np.sin(start_config.theta) + lane_width / 2 * np.sin(start_config.theta + np.pi / 2)
        )
        end_config = SE2Transform([end_config_x, end_config_y], start_config.theta)
        path = LR_path(start_config, end_config, radius)
    else:
        # Compute path from the current lane to the left lane
        end_config_y = start_config.p[1] - 2 * (
            half_point_distance * np.sin(start_config.theta) + lane_width / 2 * np.sin(start_config.theta + np.pi / 2)
        )
        end_config = SE2Transform([end_config_x, end_config_y], start_config.theta)
        path = RL_path(start_config, end_config, radius)

    path = Path(path)

    return compute_vehicle_states(path, start_speed, end_speed, radius, wheelbase)


def LR_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle = calculate_turning_circles(start_config, radius).left
    end_circle = calculate_turning_circles(end_config, radius).right

    # Compute the middle point of the curves
    middle_point_x = (start_config.p[0] + end_config.p[0]) / 2
    middle_point_y = (start_config.p[1] + end_config.p[1]) / 2
    theta = (
        tan_computation(
            end_circle.center.p[0] - start_circle.center.p[0], end_circle.center.p[1] - start_circle.center.p[1]
        )
        + np.pi / 2
    )
    middle_point = SE2Transform([middle_point_x, middle_point_y], theta)

    # Set the end config in the first cirlce
    start_circle.end_config = middle_point

    # Set the start config of the second circle
    end_circle.start_config = middle_point

    # Set the angles for the circles
    for circle in [start_circle, end_circle]:
        set_circle_angle(circle)

    # Set the streight line
    # Compute the length of the curves
    length = start_circle.length
    end_point = [
        end_circle.end_config.p[0] + np.cos(end_circle.end_config.theta) * length,
        end_circle.end_config.p[0] + np.sin(end_circle.end_config.theta) * length,
    ]
    final_lane = Line(start_circle.end_config, SE2Transform(end_point, end_circle.end_config.theta))

    return [start_circle, end_circle, final_lane]


def RL_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle = calculate_turning_circles(start_config, radius).right
    end_circle = calculate_turning_circles(end_config, radius).left

    # Compute the middle point of the curves
    middle_point_x = (start_config.p[0] + end_config.p[0]) / 2
    middle_point_y = (start_config.p[1] + end_config.p[1]) / 2
    theta = (
        tan_computation(
            end_circle.center.p[0] - start_circle.center.p[0], end_circle.center.p[1] - start_circle.center.p[1]
        )
        - np.pi / 2
    )
    middle_point = SE2Transform([middle_point_x, middle_point_y], theta)

    # Set the end config in the first cirlce
    start_circle.end_config = middle_point

    # Set the start config of the second circle
    end_circle.start_config = middle_point

    # Set the angles for the circles
    for circle in [start_circle, end_circle]:
        set_circle_angle(circle)

    length = start_circle.length
    end_point = [
        end_circle.end_config.p[0] + np.cos(end_circle.end_config.theta) * length,
        end_circle.end_config.p[1] + np.sin(end_circle.end_config.theta) * length,
    ]
    final_lane = Line(end_circle.end_config, SE2Transform(end_point, end_circle.end_config.theta))

    return [start_circle, end_circle, final_lane]


def set_circle_angle(circle: Curve):
    radius_1 = circle.start_config.p - circle.center.p
    radius_2 = circle.end_config.p - circle.center.p
    cos_ang = np.dot(radius_1, radius_2) / (np.linalg.norm(radius_1) * np.linalg.norm(radius_2))

    cos_ang = correct_approximation(cos_ang, 1.0)
    cos_ang = correct_approximation(cos_ang, -1.0)

    arc_angle = math.acos(cos_ang)
    cross = np.cross(radius_1, radius_2)
    if (cross < 0 and circle.type == DubinsSegmentType.LEFT) or (cross > 0 and circle.type == DubinsSegmentType.RIGHT):
        arc_angle = 2 * np.pi - arc_angle
    circle.arc_angle = arc_angle


def tan_computation(delta_x: float, delta_y: float) -> float:
    if delta_x != 0:
        return np.arctan2(delta_y, delta_x)
    elif delta_y > 0:
        return math.pi / 2
    else:
        return math.pi * 1.5


def correct_approximation(value: float, target: float, tolerance: float = 1e-5) -> float:
    if np.isclose(value, target, atol=tolerance):
        return target
    return value
