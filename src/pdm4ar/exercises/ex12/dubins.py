from ast import List
from collections.abc import Sequence
from tracemalloc import start

from dg_commons import SE2Transform
import math

import numpy as np

from .structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> list[SE2Transform]:
        """Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


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


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> list[Line]:
    # TODO implement here your solution
    # Consider the cases of RR or LL
    if circle_start.type == circle_end.type:
        delta_x, delta_y = circle_end.center.p - circle_start.center.p
        if delta_x == 0 and delta_y == 0:
            return []
        dir_OH = np.array([-delta_y, delta_x]) * 1 / np.sqrt(delta_y**2 + delta_x**2)

        if circle_start.type == DubinsSegmentType.LEFT:
            dir_OH = -dir_OH
        theta = tan_computation(delta_x, delta_y)
        H = SE2Transform(circle_start.center.p + circle_start.radius * dir_OH, theta)
        H_first = SE2Transform(H.p + circle_end.center.p - circle_start.center.p, theta)  # type: ignore
        tangent = [Line(H, H_first)]

    # Consider the cases RL or LR
    elif np.linalg.norm(circle_start.center.p - circle_end.center.p) > 2 * circle_end.radius:
        centers_line = circle_end.center.p - circle_start.center.p
        len_OC = np.linalg.norm(centers_line) / 2
        gamma = tan_computation(centers_line[0], centers_line[1])

        # Considering different tangentes if start circle is RIGHT or LEFT
        if circle_start.type == DubinsSegmentType.RIGHT:
            theta = math.acos(circle_start.radius / len_OC) + gamma
            ang = theta - math.pi / 2
        else:
            theta = -math.acos(circle_start.radius / len_OC) + gamma
            ang = theta + math.pi / 2

        dir_OH = np.array([math.cos(theta), math.sin(theta)]) * 1 / np.sqrt(math.cos(theta) ** 2 + math.sin(theta) ** 2)
        H = SE2Transform(circle_start.center.p + circle_start.radius * dir_OH, ang)
        H_first = SE2Transform(circle_end.center.p - dir_OH * circle_end.radius, ang)
        tangent = [Line(H, H_first)]

    # Consider impossible cases
    else:
        tangent = []

    return tangent


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float, inv=False) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    possible_solutions = ["LRL", "RLR", "LSL", "LSR", "RSL", "RSR"]

    min_path = [Line(start_config, start_config), Line(start_config, start_config), Line(start_config, start_config)]
    # Compute the path for each possible solution
    min_length = np.inf
    for solution in possible_solutions:
        match solution:
            case "LRL":
                # Compute path for LRL
                path = LRL_path(start_config, end_config, radius)
            case "RLR":
                # Compute path for LRL
                path = RLR_path(start_config, end_config, radius)
            case "LSL":
                # Compute path for LRL
                path = LSL_path(start_config, end_config, radius)
            case "LSR":
                # Compute path for LRL
                path = LSR_path(start_config, end_config, radius)
            case "RSL":
                # Compute path for LRL
                path = RSL_path(start_config, end_config, radius)
            case "RSR":
                # Compute path for LRL
                path = RSR_path(start_config, end_config, radius)

        # Compute length
        length = 0
        if path:
            for segment in path:
                length += segment.length
        else:
            length = np.inf

        # Take the path with the minimum length
        if length < min_length:
            min_length = length
            min_path = path

    # If the path is inverse, inverte the list and set the gear in the opposite sense
    if inv:
        for segment in min_path:
            segment.gear = Gear.REVERSE
            init = segment.start_config
            segment.start_config = segment.end_config
            segment.end_config = init
        min_path = min_path[::-1]

    if not min_path:
        raise ValueError("No path found")

    min_path = Path(min_path)

    return min_path


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    dubins_path = calculate_dubins_path(start_config, end_config, radius)
    dubins_path_inv = calculate_dubins_path(end_config, start_config, radius, inv=True)

    length_dubins = 0
    for segment in dubins_path:
        length_dubins += segment.length

    length_dubins_inv = 0
    for segment in dubins_path_inv:
        length_dubins_inv += segment.length

    if length_dubins < length_dubins_inv:
        return dubins_path
    else:
        return dubins_path_inv


def LRL_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle_up = calculate_turning_circles(start_config, radius).left
    start_circle_down = calculate_turning_circles(start_config, radius).left
    end_circle_up = calculate_turning_circles(end_config, radius).left
    end_circle_down = calculate_turning_circles(end_config, radius).left

    if np.linalg.norm(start_config.p - end_config.p) > 4 * radius:
        return []

    if np.array_equal(start_circle_up.center.p, end_circle_up.center.p):
        curve = Curve.create_circle(start_circle_up.center, start_config, radius, DubinsSegmentType.LEFT)
        curve.end_config = end_config
        curve_2 = Curve.create_circle(end_circle_up.center, end_config, radius, DubinsSegmentType.RIGHT)
        curve_3 = Curve.create_circle(end_circle_up.center, end_config, radius, DubinsSegmentType.LEFT)
        pathhh = [curve, curve_2, curve_3]
        for circle in pathhh:
            set_circle_angle(circle)
        return pathhh

    middle_circle_up, middle_circle_down = calculate_middle_circle(start_circle_up, end_circle_up)

    # Computations for middle_circle_up
    # Set end config of start circle
    start_circle_up.end_config = middle_circle_up.start_config

    # Set start config of end circle
    end_circle_up.start_config = middle_circle_up.end_config

    # Set the angles for the circles
    path_up = [start_circle_up, middle_circle_up, end_circle_up]
    for circle in path_up:
        set_circle_angle(circle)

    # Compute length
    length_up = 0
    if path_up:
        for segment in path_up:
            length_up += segment.length
    else:
        length_up = np.inf

    # Computations for middle_circle_down
    # Set end config of start circle
    start_circle_down.end_config = middle_circle_down.start_config

    # Set start config of end circle
    end_circle_down.start_config = middle_circle_down.end_config

    # Set the angles for the circles
    path_down = [start_circle_down, middle_circle_down, end_circle_down]
    for circle in path_down:
        set_circle_angle(circle)

    # Compute length
    length_down = 0
    if path_down:
        for segment in path_down:
            length_down += segment.length
    else:
        length_down = np.inf

    if length_up < length_down:
        return path_up
    else:
        return path_down


def RLR_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle_up = calculate_turning_circles(start_config, radius).right
    start_circle_down = calculate_turning_circles(start_config, radius).right
    end_circle_up = calculate_turning_circles(end_config, radius).right
    end_circle_down = calculate_turning_circles(end_config, radius).right

    if np.linalg.norm(start_config.p - end_config.p) > 4 * radius:
        return []

    if np.array_equal(start_circle_up.center.p, end_circle_up.center.p):
        curve = Curve.create_circle(start_circle_up.center, start_config, radius, DubinsSegmentType.RIGHT)
        curve.end_config = end_config
        curve_2 = Curve.create_circle(end_circle_up.center, end_config, radius, DubinsSegmentType.LEFT)
        curve_3 = Curve.create_circle(end_circle_up.center, end_config, radius, DubinsSegmentType.RIGHT)
        pathhh = [curve, curve_2, curve_3]
        for circle in pathhh:
            set_circle_angle(circle)
        return pathhh

    middle_circle_up, middle_circle_down = calculate_middle_circle(start_circle_up, end_circle_up)

    # Computations for middle_circle_up
    # Set end config of start circle
    start_circle_up.end_config = middle_circle_up.start_config

    # Set start config of end circle
    end_circle_up.start_config = middle_circle_up.end_config

    # Set the angles for the circles
    path_up = [start_circle_up, middle_circle_up, end_circle_up]
    for circle in path_up:
        set_circle_angle(circle)

    # Compute length
    length_up = 0
    if path_up:
        for segment in path_up:
            length_up += segment.length
    else:
        length_up = np.inf

    # Computations for middle_circle_down
    # Set end config of start circle
    start_circle_down.end_config = middle_circle_down.start_config

    # Set start config of end circle
    end_circle_down.start_config = middle_circle_down.end_config

    # Set the angles for the circles
    path_down = [start_circle_down, middle_circle_down, end_circle_down]
    for circle in path_down:
        set_circle_angle(circle)

    # Compute length
    length_down = 0
    if path_down:
        for segment in path_down:
            length_down += segment.length
    else:
        length_down = np.inf

    if length_up < length_down:
        return path_up
    else:
        return path_down


def LSL_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle = calculate_turning_circles(start_config, radius).left
    end_circle = calculate_turning_circles(end_config, radius).left
    tangents = calculate_tangent_btw_circles(start_circle, end_circle)
    if tangents:
        line = tangents[0]
    else:
        return []

    # Set the end config in the first cirlce
    start_circle.end_config = line.start_config

    # Set the start config of the second circle
    end_circle.start_config = line.end_config

    # Set the angles for the circles
    for circle in [start_circle, end_circle]:
        set_circle_angle(circle)

    return [start_circle, line, end_circle]


def LSR_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle = calculate_turning_circles(start_config, radius).left
    end_circle = calculate_turning_circles(end_config, radius).right
    tangents = calculate_tangent_btw_circles(start_circle, end_circle)
    if tangents:
        line = tangents[0]
    else:
        return []

    # Set the end config in the first cirlce
    start_circle.end_config = line.start_config

    # Set the start config of the second circle
    end_circle.start_config = line.end_config

    # Set the angles for the circles
    for circle in [start_circle, end_circle]:
        set_circle_angle(circle)

    return [start_circle, line, end_circle]


def RSL_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle = calculate_turning_circles(start_config, radius).right
    end_circle = calculate_turning_circles(end_config, radius).left
    tangents = calculate_tangent_btw_circles(start_circle, end_circle)
    if tangents:
        line = tangents[0]
    else:
        return []

    # Set the end config in the first cirlce
    start_circle.end_config = line.start_config

    # Set the start config of the second circle
    end_circle.start_config = line.end_config

    # Set the angles for the circles
    for circle in [start_circle, end_circle]:
        set_circle_angle(circle)

    return [start_circle, line, end_circle]


def RSR_path(start_config: SE2Transform, end_config: SE2Transform, radius: float):
    start_circle = calculate_turning_circles(start_config, radius).right
    end_circle = calculate_turning_circles(end_config, radius).right
    tangents = calculate_tangent_btw_circles(start_circle, end_circle)
    if tangents:
        line = tangents[0]
    else:
        return []

    # Set the end config in the first cirlce
    start_circle.end_config = line.start_config

    # Set the start config of the second circle
    end_circle.start_config = line.end_config

    # Set the angles for the circles
    for circle in [start_circle, end_circle]:
        set_circle_angle(circle)

    return [start_circle, line, end_circle]


def calculate_middle_circle(start_circle: Curve, end_circle: Curve) -> list[Curve]:
    center_end_circle = end_circle.center.p
    center_start_circle = start_circle.center.p
    radius = start_circle.radius
    dir_circles = start_circle.type

    # Calculate direction of centers_line axis
    centers_line = center_end_circle - center_start_circle
    dir_OOfirst = centers_line / np.linalg.norm(centers_line)
    dir_COsecond = np.array([dir_OOfirst[1], -dir_OOfirst[0]])
    dir_COsecond = dir_COsecond / np.linalg.norm(dir_COsecond)

    # Find centers of middle circles
    C = center_start_circle + dir_OOfirst * np.linalg.norm(centers_line) / 2
    dist = np.sqrt((2 * radius) ** 2 - (np.linalg.norm(centers_line) / 2) ** 2)
    center_middle_circle_up = SE2Transform(C + dir_COsecond * dist, 0)
    center_middle_circle_down = SE2Transform(C - dir_COsecond * dist, 0)

    # Set circle direction
    if dir_circles == DubinsSegmentType.LEFT:
        dir_circ = DubinsSegmentType.RIGHT
    else:
        dir_circ = DubinsSegmentType.LEFT

    # Find tangent point between middle_up and end circles
    dir_OfirstOsecond_up = center_middle_circle_up.p - center_end_circle
    dir_OfirstOsecond_up = dir_OfirstOsecond_up / np.linalg.norm(dir_OfirstOsecond_up)
    ang_tang_point = tan_computation(dir_OfirstOsecond_up[0], dir_OfirstOsecond_up[1]) - math.pi / 2
    if dir_circ == DubinsSegmentType.RIGHT:
        ang_tang_point += math.pi
    tang_point_up = SE2Transform(tuple(center_end_circle + dir_OfirstOsecond_up * radius), ang_tang_point)

    # Find tangent point between middle_down and end circles
    dir_OfirstOsecond_down = center_middle_circle_down.p - center_end_circle
    dir_OfirstOsecond_down = dir_OfirstOsecond_down / np.linalg.norm(dir_OfirstOsecond_down)
    ang_tang_point = tan_computation(dir_OfirstOsecond_down[0], dir_OfirstOsecond_down[1]) - math.pi / 2
    if dir_circ == DubinsSegmentType.RIGHT:
        ang_tang_point += math.pi
    tang_point_down = SE2Transform(tuple(center_end_circle + dir_OfirstOsecond_down * radius), ang_tang_point)

    # Define the two middle cirlces
    middle_circle_up = Curve.create_circle(center_middle_circle_up, tang_point_up, radius, dir_circ)
    middle_circle_down = Curve.create_circle(center_middle_circle_down, tang_point_down, radius, dir_circ)

    # Find tangent point between middle_up and start circles
    dir_OOsecond = center_middle_circle_up.p - center_start_circle
    dir_OOsecond = dir_OOsecond / np.linalg.norm(dir_OOsecond)
    ang_tang_point = tan_computation(dir_OOsecond[0], dir_OOsecond[1]) - math.pi / 2
    if dir_circ == DubinsSegmentType.RIGHT:
        ang_tang_point += math.pi
    tang_point_up = SE2Transform(tuple(center_start_circle + dir_OOsecond * radius), ang_tang_point)

    # Find tangent point between middle_down and start circles
    dir_OOsecond = center_middle_circle_down.p - center_start_circle
    dir_OOsecond = dir_OOsecond / np.linalg.norm(dir_OOsecond)
    ang_tang_point = tan_computation(dir_OOsecond[0], dir_OOsecond[1]) - math.pi / 2
    if dir_circ == DubinsSegmentType.RIGHT:
        ang_tang_point += math.pi
    tang_point_down = SE2Transform(tuple(center_start_circle + dir_OOsecond * radius), ang_tang_point)

    # Set start config of circles
    middle_circle_up.start_config = tang_point_up
    middle_circle_down.start_config = tang_point_down

    return [middle_circle_up, middle_circle_down]


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
