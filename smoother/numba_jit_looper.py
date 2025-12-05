# from typing import TYPE_CHECKING
import numpy as np
from .smooth_main import Smoother, InterpolateLoop
from numba import njit
import math


interp_functions: tuple[str, ...] = ("interpolate_nb",)


@njit
def find_start_index_nb(
    ycoord_list: np.ndarray, y_lastid: int, lower_limit_y: float
) -> int:
    """Binary search to find the starting index for Y coordinate."""
    low = 0
    high = y_lastid
    # high = len(self.ycoord_list) - 1
    while low <= high:
        mid = (low + high) // 2
        mid_y = ycoord_list[mid]  # Y coordinate
        if mid_y < lower_limit_y:
            low = mid + 1
        else:
            high = mid - 1
    return low


@njit
def find_locations_in_radius(
    current_grid_x: float,
    current_grid_y: float,
    search_radius: float,
    search_radius_squared: float,
    data_points_sorted: np.ndarray,
    with_population: bool,
    ycoord_list: np.ndarray,
    y_lastid: int,
) -> list[list[float]]:
    # Perform search
    # Before calculating distances get locations in square 2*radius²
    applicable_locations: list[list[float]] = []

    # caching limits
    limit_y = current_grid_y + search_radius
    lower_limit_y = current_grid_y - search_radius

    # Find starting index using binary search to improve efficiency
    # start_index = self.find_start_index(lower_limit_y)
    start_index = find_start_index_nb(ycoord_list, y_lastid, lower_limit_y)
    # start_index = 0
    for i in range(start_index, len(data_points_sorted)):
        datap = data_points_sorted[i]
        x = datap[1]
        y = datap[2]
        if y <= limit_y:  # removeing unnecessary if statements (-0.5s)
            if (
                # lower_limit_y <= y <= limit_y and # checked by binary search
                # abs(current_grid_y - y) <= self.search_radius and
                abs(current_grid_x - x) <= search_radius
            ):
                # Calculate squared distance x * x faster than ** 2
                dx = current_grid_x - x
                dy = current_grid_y - y
                distance_sq = dx * dx + dy * dy

                if distance_sq <= search_radius_squared:
                    # Build a tuple including distance_sq; avoid mutating datap.append() which returns None
                    if with_population:
                        # datap: [id, x, y, population, rate]
                        applicable_locations.append(
                            [datap[0], x, y, datap[3], datap[4], distance_sq]
                        )
                    else:
                        # datap: [id, x, y, rate]
                        applicable_locations.append(
                            [datap[0], x, y, datap[3], distance_sq]
                        )
        else:
            # Since data is sorted by Y break early
            break
    return applicable_locations


@njit
def interpolate_nb(
    locations_in_radius: list[list[float]],
    half_distance_squared: float,
    with_population: bool,
) -> float:
    """Interpolate smoothed rate using distance weighted interpolation."""
    total_weight = 0.0
    weighted_rate_sum = 0.0
    if with_population:
        total_population = 0.0
        for point_data in locations_in_radius:
            rate, dist_sq, population = point_data[4], point_data[5], point_data[3]
            weight = 1.0 / (1.0 + dist_sq / half_distance_squared)
            total_weight += weight * population
            weighted_rate_sum += rate * weight * population
            total_population += population
        if total_weight > 0.0:
            smoothed_rate = weighted_rate_sum / total_weight
        else:
            smoothed_rate = 0.0
    else:
        for point_data in locations_in_radius:
            rate, dist_sq = point_data[3], point_data[4]
            weight = 1.0 / (1.0 + dist_sq / half_distance_squared)
            total_weight += weight
            weighted_rate_sum += rate * weight
        if total_weight > 0.0:
            smoothed_rate = weighted_rate_sum / total_weight
        else:
            smoothed_rate = 0.0
    return smoothed_rate


@njit
def loop_with_numba(
    point_data: list[list[float]],
    grid_settings_rows: int,
    grid_settings_cols: int,
    grid_size: float,
    half_distance_squared: float,
    search_radius_squared: int,
    interpolation_function_name: str,
) -> np.ndarray:
    # search_window = search_window_with_numba(point_data, search_radius_squared)

    # caching variables
    search_radius = math.sqrt(search_radius_squared)
    y_lastid = len(point_data) - 1
    # y_coord_list = [loc[2] for loc in point_data]
    y_coord_list = np.array([loc[2] for loc in point_data], dtype=np.float64)
    point_data_np = np.array(point_data, dtype=np.float64)

    with_population = len(point_data[0]) == 5  # with population data

    # Use numpy array instead of list for results to avoid Numba type inference errors
    num_cells = grid_settings_rows * grid_settings_cols
    result_array = np.zeros((num_cells, 3), dtype=np.float64)

    for row in range(grid_settings_rows):
        y_coord = row * grid_size + (grid_size / 2)
        for col in range(grid_settings_cols):
            x_coord = col * grid_size + (grid_size / 2)
            # Find locations in radius
            locations_in_radius = find_locations_in_radius(
                x_coord,
                y_coord,
                search_radius,
                search_radius_squared,
                point_data_np,
                with_population,
                y_coord_list,
                y_lastid,
            )

            # Interpolate
            # interpolate_function = eval(interpolation_function_name)
            if interpolation_function_name == "interpolate_nb":
                smoothed_rate: float = interpolate_nb(
                    locations_in_radius, half_distance_squared, with_population
                )
            else:
                smoothed_rate = 0.0  # Fallback, should not happen

            # Store result
            index = row * grid_settings_cols + col
            result_array[index, 0] = x_coord
            result_array[index, 1] = y_coord
            result_array[index, 2] = smoothed_rate

    return result_array


@njit
def prepare_xcoord_list_numba(grid_size: float, ncols: int) -> np.ndarray:
    """Prepare numpy array of x coordinates for grid columns using Numba."""
    cell_center = grid_size / 2
    column_coord_array = np.empty(ncols, dtype=np.float64)
    for column_id in range(ncols):
        column_coord_array[column_id] = column_id * grid_size + cell_center
    return column_coord_array


class NumbaJitLooper(InterpolateLoop, Smoother):
    smoothed_data: np.ndarray

    def __init__(
        self,
        point_data,
        grid_settings,
        half_distance,
        search_radius,
        interpolation_function,
    ):
        super().__init__(
            point_data,
            grid_settings,
            half_distance,
            search_radius,
            interpolation_function,
        )

    def loop_coords(self) -> None:
        """Perform smoothing using Numba JIT compiled functions."""
        # cell_center = self.grid_settings.grid_size / 2
        # column_coord_array = prepare_xcoord_list_numba(self.grid_settings.grid_size, self.grid_settings.cols)

        self.smoothed_data = loop_with_numba(
            self.point_data,
            self.grid_settings.rows,
            self.grid_settings.cols,
            self.grid_settings.grid_size,
            self.half_distance_squared,
            self.search_radius_squared,
            interpolation_function_name="interpolate_nb",
        )
