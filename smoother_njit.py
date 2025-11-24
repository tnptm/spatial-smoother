"""Module containing Numba-optimized functions for smoothing location data.
it is generated with Claude Sonnet 4 from normal python version of smoother.py.
It added normalization into smoothing weights
"""

import math
import time

import numpy as np
from numba import njit, types
from numba.typed import List

# Numba-compatible type definitions
Point = types.UniTuple(types.float64, 2)

# Using numpy arrays instead of dataclasses for numba compatibility
# LocationData: [id, x, y]
# LocationDataRate: [id, x, y, rate]
# LocationFound: [id, x, y, rate, distance]
# GridCellRate: [x, y, rate]


@njit
def create_location_data(id_val: int, x: float, y: float):
    """Create location data as numpy array [id, x, y]"""
    return np.array([id_val, x, y], dtype=np.float64)


@njit
def create_location_data_rate(id_val: int, x: float, y: float, rate: float):
    """Create location data rate as numpy array [id, x, y, rate]"""
    return np.array([id_val, x, y, rate], dtype=np.float64)


@njit
def create_location_found(
    id_val: int, x: float, y: float, rate: float, distance: float
):
    """Create location found as numpy array [id, x, y, rate, distance]"""
    return np.array([id_val, x, y, rate, distance], dtype=np.float64)


@njit
def create_grid_cell_rate(x: float, y: float, rate: float):
    """Create grid cell rate as numpy array [x, y, rate]"""
    return np.array([x, y, rate], dtype=np.float64)


@njit
def calc_distance(
    grid_x: float, grid_y: float, location_x: float, location_y: float
) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((grid_x - location_x) ** 2 + (grid_y - location_y) ** 2)


@njit
def find_locations_in_radius(
    random_locations, current_grid_x: float, current_grid_y: float, search_radius: float
):
    """Find all locations within search radius of current grid point"""
    applicable_locations = List.empty_list(types.float64[:])

    for i in range(len(random_locations)):
        location = random_locations[i]
        location_x = location[1]  # x coordinate
        location_y = location[2]  # y coordinate

        # Quick bounding box check first
        if (
            abs(current_grid_x - location_x) <= search_radius
            and abs(current_grid_y - location_y) <= search_radius
        ):
            # Calculate actual distance
            distance = calc_distance(
                current_grid_x, current_grid_y, location_x, location_y
            )

            if distance <= search_radius:
                # Create LocationFound: [id, x, y, rate, distance]
                location_found = create_location_found(
                    location[0],  # id
                    location_x,  # x
                    location_y,  # y
                    location[3],  # rate
                    distance,  # distance
                )
                applicable_locations.append(location_found)

    return applicable_locations


@njit
def distance_weight(distance: float, half_distance: float) -> float:
    """Calculate distance weight for interpolation"""
    return 1.0 / (1.0 + (distance / half_distance) ** 2)


@njit
def interpolate_distance_weighted(window_data, half_distance: float) -> float:
    """Perform distance weighted interpolation"""
    smoothed_rate = 0.0
    total_weight = 0.0

    for i in range(len(window_data)):
        point_data = window_data[i]
        rate = point_data[3]  # rate
        distance = point_data[4]  # distance

        weight = distance_weight(distance, half_distance)
        smoothed_rate += rate * weight
        total_weight += weight

    # Normalize by total weight to avoid bias
    if total_weight > 0:
        return smoothed_rate / total_weight
    else:
        return 0.0


@njit
def smooth_grid(
    point_data,
    grid_rows: int,
    grid_cols: int,
    grid_size: float,
    half_distance: float,
    search_radius: float,
):
    """Perform smoothing on the entire grid"""

    # Pre-calculate column coordinates
    column_coords = np.zeros(grid_cols, dtype=np.float64)
    for col in range(grid_cols):
        column_coords[col] = col * grid_size

    # Initialize result list
    smoothed_data = List.empty_list(types.float64[:])

    for row in range(grid_rows):
        y_coord = row * grid_size

        for col in range(grid_cols):
            x_coord = column_coords[col]

            # Calculate grid cell center
            grid_cell_center_x = x_coord + (grid_size / 2.0)
            grid_cell_center_y = y_coord + (grid_size / 2.0)

            # Find locations within search radius
            search_window_points = find_locations_in_radius(
                point_data, grid_cell_center_x, grid_cell_center_y, search_radius
            )

            # Perform interpolation
            if len(search_window_points) > 0:
                smoothed_rate_xy = interpolate_distance_weighted(
                    search_window_points, half_distance
                )
            else:
                smoothed_rate_xy = 0.0

            # Create grid cell rate and add to results
            grid_cell_rate = create_grid_cell_rate(x_coord, y_coord, smoothed_rate_xy)
            smoothed_data.append(grid_cell_rate)

    return smoothed_data


@njit
def create_random_location_data_njit(
    grid_rows: int,
    grid_cols: int,
    grid_size: float,
    number_of_locations: int,
    seed: int = 42,
):
    """Create random locations within a grid using numba-compatible random generation"""
    np.random.seed(seed)

    rate_data = List.empty_list(types.float64[:])

    for loc_id in range(number_of_locations):
        x = np.random.random() * grid_cols * grid_size
        y = np.random.random() * grid_rows * grid_size
        rate = np.random.random() * 100.0  # max rate = 100

        location_data_rate = create_location_data_rate(loc_id + 1, x, y, rate)
        rate_data.append(location_data_rate)

    return rate_data


# Non-numba wrapper class for easier interface
class Smoother:
    """Wrapper class for the numba-compiled smoothing functions"""

    def __init__(
        self,
        point_data,
        grid_rows: int,
        grid_cols: int,
        grid_size: float,
        half_distance: float,
        search_radius: float,
    ):
        self.point_data = point_data
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_size = grid_size
        self.half_distance = half_distance
        self.search_radius = search_radius
        self.smoothed_data = None

    def smooth(self):
        """Perform smoothing using numba-compiled functions"""
        self.smoothed_data = smooth_grid(
            self.point_data,
            self.grid_rows,
            self.grid_cols,
            self.grid_size,
            self.half_distance,
            self.search_radius,
        )

    def print_results(self, max_items: int = 10):
        """Print first max_items results"""
        if self.smoothed_data is None:
            print("No smoothed data available. Run smooth() first.")
            return

        print(f"Printing first {max_items} results:")
        for i in range(min(max_items, len(self.smoothed_data))):
            cell = self.smoothed_data[i]
            print(f"X: {cell[0]:.2f} Y: {cell[1]:.2f} Rate: {cell[2]:.2f}")


def create_random_locations_wrapper(
    grid_rows: int,
    grid_cols: int,
    grid_size: float,
    number_of_locations: int,
    seed: int = 42,
):
    """Non-numba wrapper for creating random locations"""
    return create_random_location_data_njit(
        grid_rows, grid_cols, grid_size, number_of_locations, seed
    )


if __name__ == "__main__":
    # Measure time in seconds
    start_time = time.time()

    # Grid settings
    grid_rows = 1000
    grid_cols = 1000
    grid_size = 500.0
    half_distance = 10000.0
    search_radius = 150000.0
    number_of_locations = 100

    # Create random location data
    random_locations = create_random_locations_wrapper(
        grid_rows, grid_cols, grid_size, number_of_locations
    )

    # Create smoother and run
    my_smoother = Smoother(
        random_locations, grid_rows, grid_cols, grid_size, half_distance, search_radius
    )

    # Perform smoothing
    my_smoother.smooth()

    end_time = time.time()

    # Print results
    my_smoother.print_results()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
