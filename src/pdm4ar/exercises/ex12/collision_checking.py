from shapely.geometry import Point


def collision_checking(
    my_positions: list[tuple[float, float]],
    other_positions: list[tuple[float, float]],
    my_radius: float,
    other_radius: float,
) -> list[int]:
    """
    This function checks if our current trajectory is in collision with other agents' trajectories.
    :param my_positions: List of tuples containing the current agent's positions
    :param other_positions: List of tuples containing the other agents' positions
    :param my_radius: Integer indicating the radius of our agent
    :param other_radius: Integer indicating the radius of the other agents
    :return: List of integers indicating the time steps in which a collision occurs
    """
    # Check on the lenght of the lists
    if len(my_positions) == 0 or len(other_positions) == 0 or len(my_positions) != len(other_positions):
        print("ERROR! The lists for the positions are empty or have different lengths.")
        return [-1]

    # Create the shapely points buffered for our car and the other agent
    my_points = []
    for my_position in my_positions:
        my_points.append(Point(my_position[0], my_position[1]).buffer(my_radius))

    other_points = []
    for other_position in other_positions:
        other_points.append(Point(other_position[0], other_position[1]).buffer(other_radius))

    # Check if corrisponding points are in collision
    collision_indexes = []
    for my_point, other_point in zip(my_points, other_points):
        if my_point.intersects(other_point):
            collision_indexes.append(my_points.index(my_point))

    return collision_indexes
