from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate
from dg_commons import SE2Transform


class CollisionChecker:
    def __init__(self, portion_of_trajectory) -> None:
        self.geometries = {}
        self.portion_of_trajectory = portion_of_trajectory

    def add_other_geometry(self, name, other_geometry):
        self.geometries[name] = other_geometry

    def collision_checking(
        self,
        my_positions: list[SE2Transform],
        my_name: str,
        other_positions_dict: dict[str, list[SE2Transform]],
    ) -> dict[str, list[int]]:
        """
        This function checks if our current trajectory is in collision with other agents' trajectories.
        :param my_positions: List of tuples containing the current agent's positions
        :param other_positions: List of tuples containing the other agents' positions
        :param my_radius: Integer indicating the radius of our agent
        :param other_radius: Integer indicating the radius of the other agents
        :param portion_of_trajectory: Float indicating the portion of the trajectory to consider for the collision checking
        :return: List of integers indicating the time steps in which a collision occurs
        """
        self.my_name = my_name

        # Check on the lenght of the lists
        if len(my_positions) == 0 or all(
            len(my_positions) // self.portion_of_trajectory != len(other_positions)
            for other_positions in other_positions_dict.values()
        ):
            print("ERROR! The lists for the positions are empty or have different lengths.")
            return {}

        # Create the shapely points buffered for our car and the other agent
        my_points = []
        for my_position in my_positions[0 : (len(my_positions) // self.portion_of_trajectory)]:
            my_points.append(self.compute_car_position(my_position, self.my_name))

        collision_indexes = {}
        for other_name, other_positions in other_positions_dict.items():
            other_points = []
            for other_position in other_positions:
                other_points.append(self.compute_car_position(other_position, other_name))

            # Check if corrisponding points are in collision
            collision_indexes[other_name] = []
            for my_point, other_point in zip(my_points, other_points):
                if my_point.intersects(other_point):
                    collision_indexes[other_name].append(my_points.index(my_point))

        """self.plot_positions_with_radius(
            my_positions[0 : (len(my_positions) // self.portion_of_trajectory)],
            other_positions,
        )"""

        return collision_indexes

    def compute_car_position(self, position: SE2Transform, name: str) -> Polygon:
        """
        This function computes the car position given the current position and the car geometry.
        :param position: Tuple containing the current position of the car
        :return: Shapely polygon representing the car around the given position
        """
        x, y = position.p
        theta = position.theta
        geometry = self.geometries[name]

        if isinstance(geometry, Polygon):
            # Translate the car geometry to the current position
            current_center = geometry.centroid
            translation = (x - current_center.x, y - current_center.y)
            translated_geometry = translate(geometry, xoff=translation[0], yoff=translation[1])

        # Consider the case of my car
        else:
            # Rectangle vertices
            half_width, half_height = geometry.w_half, geometry.length / 2
            vertices = [
                (-half_width, -half_height),  # In basso a sinistra
                (half_width, -half_height),  # In basso a destra
                (half_width, half_height),  # In alto a destra
                (-half_width, half_height),  # In alto a sinistra
            ]

            # Rotate the vertices
            rotated_vertices = []
            for dx, dy in vertices:
                x_rot = x + (dx * np.cos(theta) - dy * np.sin(theta))
                y_rot = y + (dx * np.sin(theta) + dy * np.cos(theta))
                rotated_vertices.append((x_rot, y_rot))

            # Translate the car geometry to the current position
            translated_geometry = translate(Polygon(rotated_vertices), xoff=x, yoff=y)

        return translated_geometry

    def plot_positions_with_radius(
        self,
        my_positions: list[tuple[float, float]],
        my_radius: float,
        other_positions: list[tuple[float, float]],
        other_radius: float,
    ):
        """
        This function plots the positions on the map with the given radius.
        :param my_positions: List of tuples containing the current agent's positions
        :param my_radius: Float indicating the radius of the circles to plot for the current agent
        :param other_positions: List of tuples containing the other agents' positions
        :param other_radius: Float indicating the radius of the circles to plot for the other agents
        """
        fig, ax = plt.subplots()
        for position in my_positions:
            point = Point(position[0], position[1])
            circle = point.buffer(my_radius)
            x, y = circle.exterior.xy
            ax.plot(x, y, color="blue")
            ax.plot(position[0], position[1], "o", color="blue")

        for position in other_positions:
            point = Point(position[0], position[1])
            circle = point.buffer(other_radius)
            x, y = circle.exterior.xy
            ax.plot(x, y, color="red")
            ax.plot(position[0], position[1], "o", color="red")

        ax.set_aspect("equal", "box")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Positions with Radius")
        plt.grid(True)
        plt.savefig("positions_with_radius.png")
