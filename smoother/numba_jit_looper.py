"""
smoother.numba_jit_looper
-------------------------

Numba-accelerated spatial smoothing utilities and an InterpolateLoop/Smoother
implementation that computes a distance-weighted smoothed rate on a regular
grid. The functions in this module are designed to be JIT compiled with Numba
(@njit) and therefore impose simple, statically-typed data-layout requirements.

Key concepts
- Input point_data should be sorted by Y coordinate (ascending) to enable
    efficient early-exit during radius search and to allow binary-search
    for the starting Y index.
- Two point record formats are supported:
        * Without population:  [id, x, y, rate]
        * With population:     [id, x, y, population, rate]
    The code detects which format is in use by inspecting the length of the
    first record.
- All numeric arrays/values used by njit functions should be numpy arrays
    with float64 dtype (or scalars convertible to float64). Python lists of
    lists are accepted at the Python boundary but are converted to numpy arrays
    inside the JIT entrypoint.

Provided functions and class
- find_start_index_nb(ycoord_list, y_lastid, lower_limit_y) -> int
        Binary search to find the lowest index in ycoord_list whose Y >=
        lower_limit_y. ycoord_list is a 1-D numpy array of Y coordinates.
        Returns an integer index (0..y_lastid+1).

- find_locations_in_radius(current_grid_x, current_grid_y, search_radius,
                                                     search_radius_squared, data_points_sorted,
                                                     with_population, ycoord_list, y_lastid) -> list[list[float]]
        Scans point data starting from the binary-searched Y start index and
        collects points within the circular search radius. Uses an axis-aligned
        square pre-filter on X and Y before computing squared Euclidean distance.
        Returns a typed Numba list-of-lists where each inner list encodes a
        point's data plus the distance squared:
            * with_population == True:
                    [id, x, y, population, rate, distance_sq]
            * with_population == False:
                    [id, x, y, rate, distance_sq]
        Note: data_points_sorted must be sorted by Y increasing.

- interpolate_nb(locations_in_radius, half_distance_squared, with_population) -> float
        Compute a distance-weighted interpolated (smoothed) rate from the list of
        nearby points. Uses the weight function:
                weight = 1 / (1 + dist_sq / half_distance_squared)
        If with_population is True then weights are multiplied by population and
        the weighted sum is normalized by the total (weight * population). Returns
        0.0 if no points are available or total weight is zero.

- loop_with_numba(point_data, grid_settings_rows, grid_settings_cols, grid_size,
                                    half_distance_squared, search_radius_squared,
                                    interpolation_function_name) -> np.ndarray
        Numba entrypoint that performs the full grid loop. Behaviour:
            * Converts point_data into numpy float64 arrays and builds a y-coordinate
                array for binary searching.
            * Computes the grid cell center coordinates and for each cell:
                    - Finds nearby points via find_locations_in_radius
                    - Calls interpolate_nb (selected by interpolation_function_name)
                    - Stores results in a (rows*cols, 3) numpy array: [x, y, smoothed_rate]
        Inputs should be numeric and sized appropriately. The function returns a
        float64 numpy array shaped (rows*cols, 3).

- prepare_xcoord_list_numba(grid_size, ncols) -> np.ndarray
        Small helper that returns a 1-D numpy array of column center X coordinates
        for ncols columns using a grid cell center offset (grid_size / 2).

- NumbaJitLooper(InterpolateLoop, Smoother)
        Class wrapper that exposes the loop_coords() method to run the smoothing
        pipeline using loop_with_numba. After execution, `self.smoothed_data` is a
        numpy array with one row per grid cell and columns [x_center, y_center,
        smoothed_rate].

Usage notes
- Because these routines are JIT compiled, types must be consistent between
    calls. Convert or validate inputs (point formatting, dtypes, sorting) in the
    Python layer before calling the njit entrypoint to avoid hard-to-debug type
    inference errors.
- The module avoids Python features that Numba cannot compile (e.g. eval,
    complex object attribute access). Keep interpolation and distance metrics
    simple and vectorizable where possible.
- Performance considerations:
        * Pre-sorting points by Y and using binary search yields an early-exit and
            generally reduces the number of distance computations.
        * Using squared distances avoids expensive sqrt calls during neighbor search.
        * Passing numpy arrays of dtype float64 and keeping data in contiguous
            layouts helps Numba optimize loops.

Examples
- Typical flow (Python-side, conceptual):
        1. Prepare point_data as a list of lists sorted by Y.
        2. Create grid settings: rows, cols, grid_size.
        3. Instantiate NumbaJitLooper(point_data, grid_settings, half_distance,
             search_radius, interpolation_function="interpolate_nb").
        4. Call looper.loop_coords().
        5. Read looper.smoothed_data (shape = rows*cols x 3).

Limitations & Edge Cases
- The binary search assumes ycoord_list is ascending and y_lastid points to
    the last valid index.
- If all weights are zero (e.g. no neighbors), interpolator returns 0.0.
- The module is specialized for 2D Cartesian coordinates and Euclidean
    distance; extension to spherical coords or anisotropic metrics would require
    changes to search and weighting logic.


"""
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
    """Find locations within the search radius from current grid cell center."""
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
    """Numba JIT compiled smoothing loop over grid cells."""
    

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
    """Smoother using Numba JIT compiled loop for distance-weighted smoothing."""
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
