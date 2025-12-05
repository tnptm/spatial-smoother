"""
Docstring for smoother.smooth_main
This module contains classes and functions for smoothing spatial data using distance-based interpolation methods.
It includes definitions for grid maps, search windows, interpolation methods, and the main Smoother class.


author: Toni Patama tonipat047@gmail.com, 2025-11-28, version 1.0
"""

from datetime import datetime
import math

# import sys
# import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

# from random import random
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
# from numba_jit_looper import NumbaJitLooper, loop_with_numba

Point = tuple[float, float]


@dataclass
class LocationData:
    id: int
    point: Point


@dataclass
class LocationDataRate:
    location_data: LocationData
    rate: float


@dataclass
class LocationFound:
    point_data: LocationDataRate
    distance: float


@dataclass
class GridMapDefinition:
    rows: int
    cols: int
    grid_size: float


class SearchWindow:
    """Search window for finding locations within a certain radius.

    Args:
        random_locations: List of random locations (int,x,y,population,rate).
        current_grid_point: Current grid point.
        search_radius: Search radius.

    Returns:
        List of found locations within the search radius.
    """

    # random_locations: list[LocationDataRate]
    data_points_sorted: list[list[float]]
    search_radius_squared: int

    def __init__(
        self,
        data_points_sorted: list[list[float]],
        search_radius_squared: int,
    ):
        self.data_points_sorted = data_points_sorted
        self.search_radius_squared = search_radius_squared
        self.search_radius = math.sqrt(search_radius_squared)
        # global sw_data
        # sw_data = np.array([ y_coord[2] for y_coord in data_points_sorted ])  # for numba njit function use
        self.ycoord_list = [
            y_coord[2] for y_coord in data_points_sorted
        ]  # for numba njit function use
        self.y_lastid = len(self.ycoord_list) - 1  # cache length
        self.self_data_definition = {
            "id": 0,
            "x": 1,
            "y": 2,
            "population": 3,
            "rate": 4,
        }
        self.with_population = (
            data_points_sorted[0].__len__() == 5
        )  # with population data

    def find_start_index_nb(self, lower_limit_y: float) -> int:
        """Binary search to find the starting index for Y coordinate."""
        low = 0
        high = self.y_lastid
        # high = len(self.ycoord_list) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_y = self.ycoord_list[mid]  # Y coordinate
            if mid_y < lower_limit_y:
                low = mid + 1
            else:
                high = mid - 1
        return low

    def find_locations_in_radius(
        self, current_grid_x: float, current_grid_y: float
    ) -> list[tuple]:
        # Perform search
        # Before calculating distances get locations in square 2*radius²
        applicable_locations: list[tuple] = []

        # caching limits
        limit_y = current_grid_y + self.search_radius
        lower_limit_y = current_grid_y - self.search_radius

        # Find starting index using binary search to improve efficiency
        # start_index = self.find_start_index(lower_limit_y)
        start_index = self.find_start_index_nb(lower_limit_y)
        # start_index = 0
        for i in range(start_index, len(self.data_points_sorted)):
            datap = self.data_points_sorted[i]
            x = datap[1]
            y = datap[2]
            if y <= limit_y:  # removeing unnecessary if statements (-0.5s)
                if (
                    # lower_limit_y <= y <= limit_y and # checked by binary search
                    # abs(current_grid_y - y) <= self.search_radius and
                    abs(current_grid_x - x) <= self.search_radius
                ):
                    # Calculate squared distance x * x faster than ** 2
                    dx = current_grid_x - x
                    dy = current_grid_y - y
                    distance_sq = dx * dx + dy * dy

                    if distance_sq <= self.search_radius_squared:
                        # Build a tuple including distance_sq; avoid mutating datap.append() which returns None
                        if self.with_population:
                            # datap: [id, x, y, population, rate]
                            applicable_locations.append(
                                (datap[0], x, y, datap[3], datap[4], distance_sq)
                            )
                        else:
                            # datap: [id, x, y, rate]
                            applicable_locations.append(
                                (datap[0], x, y, datap[3], distance_sq)
                            )
            else:
                # Since data is sorted by Y break early
                break
        return applicable_locations


class Interpolator(ABC):
    """Base class for interpolation methods."""

    @abstractmethod
    def interpolate(self, *args: Any, **kwargs: Any) -> float:
        # Perform interpolation
        raise NotImplementedError


class DistanceWeightedInterpolator(Interpolator):
    # def __init__(self)#, window_data: list[LocationFound]):
    #    self.window_data: list[LocationFound] = window_data

    @staticmethod
    def distance_weight_sq(
        distance_squared: float, half_distance_squared: float
    ) -> float:
        """Calculate distance weight based on squared distance."""
        # Optional alternative method using distance instead of squared distance
        # @staticmethod
        # def distance_weight(self, distance: float, half_distance: float) -> float:
        #     return 1 / (1 + (distance / half_distance)**2 )
        return 1 / (1 + (distance_squared / half_distance_squared))

    def interpolate(
        self,
        window_data: list[tuple[float, float, float, float, float]],
        half_distance_squared: float,
    ) -> float:
        """Interpolate the rate based on distance-weighted averaging.

        Args:
            half_distance_squared: The half-distance squared used for weighting.
            window_data: List of tuples containing LocationDataRate and its squared distance.

        Returns:
            The interpolated rate.
        """
        smoothed_rate = 0
        total_weights = 0
        if not window_data:
            return 0.0
        for point_data in window_data:
            # Perform interpolation using point_data
            # dist_weight = self.distance_weight_sq(point_data[-1], half_distance_squared)
            dist_weight = 1 / (
                1 + (point_data[4] / half_distance_squared)
            )  # inlined distance weight_sq()
            smoothed_rate += dist_weight * point_data[3]
            total_weights += dist_weight

        if total_weights == 0:
            return 0
        weighted_mean = smoothed_rate / total_weights

        return weighted_mean


class DistancePopulationWeightedInterpolator(Interpolator):
    def interpolate(
        self,
        window_data: list[tuple[int, float, float, float, float, float]],
        half_distance_squared: float,
    ) -> float:
        """Interpolate the rate based on distance-weighted averaging.

        Args:
            half_distance_squared: The half-distance squared used for weighting.
            window_data: List of tuples containing LocationDataRate and its squared distance.
                (index, x, y, population, rate, distance_squared)

        Returns:
            The interpolated rate.
        """
        smoothed_rate = 0
        total_weights = 0

        population_index = 3  # index of population in the tuple
        distance_index = 5  # index of distance_squared in the tuple
        rate_index = 4  # index of rate in the tuple
        if not window_data:
            return 0.0
        for point_data in window_data:
            # Perform interpolation using point_data

            # distance-population weighting Wdist * Wpop
            dist_pop_weight_i = point_data[population_index] / (
                1 + (point_data[distance_index] / half_distance_squared)
            )  # inlined distance weight_sq()
            # population_weight = point_data[3]
            smoothed_rate += dist_pop_weight_i * point_data[rate_index]
            total_weights += dist_pop_weight_i

        if total_weights == 0:
            return 0
        weighted_mean = smoothed_rate / total_weights

        return weighted_mean


@dataclass
class GridCellRate:
    point: Point
    rate: float


class InterpolateLoop(ABC):
    @abstractmethod
    def loop_coords(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


class Smoother:
    """Smoother class for smoothing data using search window data."""

    # data: list[list[float]]
    point_data: list  # list[LocationDataRate]
    # smoothed_data: list[GridCellRate]
    grid_settings: GridMapDefinition
    half_distance: int
    search_radius: int
    interpolation_function: Interpolator

    def __init__(
        self,
        # data: list[list[float]],
        point_data: list,  # list[LocationDataRate],
        grid_settings: GridMapDefinition,
        half_distance: int,
        search_radius: int,
        interpolation_function: Interpolator,
    ):
        # self.data = data
        self.point_data = point_data  # sorted by y coordinate before passing
        self.grid_settings = grid_settings
        self.half_distance = half_distance
        self.half_distance_squared = half_distance**2
        self.search_radius = search_radius
        # self.interpolation_function = interpolation_function
        self.smoothed_data = []
        self.search_radius_squared = search_radius**2
        self.interpolation_function = interpolation_function

    def prepare(self):
        # Prepare data for smoothing, generate rates for each location
        pass

    def smooth_old(self, search_window: SearchWindow):
        """
        Perform smoothing using search_window_data

        It uses a search window to find nearby data points and calculates the average value within the window.

        """
        # Perform smoothing
        # pass
        ## CREATE colID,Xcoord list
        cell_center = self.grid_settings.grid_size / 2

        column_coord_list = [
            column_id * self.grid_settings.grid_size + cell_center
            for column_id in range(self.grid_settings.cols)
        ]

        # search_window = SearchWindow(self.point_data, self.search_radius_squared)
        # interpolator: DistanceWeightedInterpolator = DistanceWeightedInterpolator()

        # pre-allocate result list
        smoothed_data: list[tuple[float, float, float] | None] = [None] * (
            self.grid_settings.rows * self.grid_settings.cols
        )
        index = 0
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size + cell_center
            for x_coord in column_coord_list:
                # define search window data
                # grid_cell_center_x = x_coord + cell_center
                # grid_cell_center_y = y_coord + cell_center

                # interpolator = DistanceWeightedInterpolator(search_window.find_locations_in_radius(grid_cell_center_x, grid_cell_center_y))
                smoothed_rate_xy = self.interpolation_function.interpolate(
                    search_window.find_locations_in_radius(x_coord, y_coord),
                    self.half_distance_squared,
                )

                smoothed_data[index] = (
                    # GridCellRate(Point((x_coord, y_coord)), smoothed_rate_xy)
                    (x_coord, y_coord, smoothed_rate_xy)
                )
                index += 1

        self.smoothed_data = smoothed_data

    def smooth(
        self, looper: Callable
    ):  # , search_window: SearchWindow, looper_function: InterpolateLoop):
        """Perform smoothing using specified looper function for coordinate iteration."""
        looper()  # self, search_window, looper_function)

    def print(self, all: bool = False) -> None:
        """Print smoothed data of 10 first cells as a sample"""
        data_to_print = self.smoothed_data if all else self.smoothed_data[:10]
        print("X,Y,Rate")
        for cell in data_to_print:
            # for cell in row:
            print(
                # f"X: {cell.point[0]:.2f} Y: {cell.point[1]:.2f} Rate: {cell.rate:.2f}",
                # end=" ",
                # f"{cell.point[0]:.2f},{cell.point[1]:.2f},{cell.rate:.2f}",
                f"{cell[0]:.0f},{cell[1]:.0f},{cell[2]:.3f}",
            )
            # print()

    def save(self):
        # Save smoothed data to ascii grid file (*.asc)
        file_name = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.asc"
        with open(file_name, "w") as f:
            # Write header
            f.write(f"ncols         {self.grid_settings.cols}\n")
            f.write(f"nrows         {self.grid_settings.rows}\n")
            f.write("xllcorner     0.0\n")
            f.write("yllcorner     0.0\n")
            f.write(f"cellsize      {self.grid_settings.grid_size}\n")
            f.write("NODATA_value  -9999\n")

            # Write data row by row
            for row in range(self.grid_settings.rows):
                row_data = []
                for col in range(self.grid_settings.cols):
                    index = row * self.grid_settings.cols + col
                    rate = (
                        self.smoothed_data[index][2]
                        if self.smoothed_data[index] is not None
                        else -9999
                    )
                    row_data.append(f"{rate:.3f}")
                f.write(" ".join(row_data) + "\n")

    def plot(self, figsize: tuple[float, float] = (12, 8), dpi: int = 150):
        """Plot the smoothed data using matplotlib and make a mesh plot.
        Args:
            figsize: Figure size in inches (width, height).
            dpi: Dots per inch for the figure resolution.
        """

        x_coords = [cell[0] for cell in self.smoothed_data]
        y_coords = [cell[1] for cell in self.smoothed_data]
        rates = [cell[2] for cell in self.smoothed_data]

        # Reshape data for grid plotting
        X = np.array(x_coords).reshape(self.grid_settings.rows, self.grid_settings.cols)
        Y = np.array(y_coords).reshape(self.grid_settings.rows, self.grid_settings.cols)
        Z = np.array(rates).reshape(self.grid_settings.rows, self.grid_settings.cols)

        plt.figure(figsize=figsize, dpi=dpi)  # increase resolution/size
        plt.pcolormesh(X, Y, Z, shading="auto")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Smoothed Data Surface Plot")
        plt.colorbar(label="Rate")
        plt.tight_layout()
        plt.show()


class PurePythonLooper(InterpolateLoop, Smoother):
    # smoothed_data: list[tuple[float, float, float]|None]
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
        """Perform smoothing using pure Python looping over coordinates."""
        cell_center = self.grid_settings.grid_size / 2
        column_coord_list = [
            column_id * self.grid_settings.grid_size + cell_center
            for column_id in range(self.grid_settings.cols)
        ]

        search_window = SearchWindow(self.point_data, self.search_radius_squared)
        smoothed_data: list[tuple[float, float, float] | None] = [None] * (
            self.grid_settings.rows * self.grid_settings.cols
        )
        index = 0
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size + cell_center
            for x_coord in column_coord_list:
                # interpolator = DistanceWeightedInterpolator(search_window.find_locations_in_radius(grid_cell_center_x, grid_cell_center_y))
                smoothed_rate_xy = self.interpolation_function.interpolate(
                    search_window.find_locations_in_radius(x_coord, y_coord),
                    self.half_distance_squared,
                )

                smoothed_data[index] = (x_coord, y_coord, smoothed_rate_xy)
                index += 1

        self.smoothed_data = smoothed_data


class NumpyVectorizedLooper(InterpolateLoop, Smoother):
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

    def prepare_xcoord_list(self) -> np.ndarray:
        """Prepare numpy array of x coordinates for grid columns."""
        cell_center = self.grid_settings.grid_size / 2
        column_coord_array = np.array(
            [
                column_id * self.grid_settings.grid_size + cell_center
                for column_id in range(self.grid_settings.cols)
            ]
        )
        return column_coord_array

    # @staticmethod
    def numpy_loop_coords(
        self, xy_coords: np.ndarray, search_window: SearchWindow
    ) -> None:
        """Perform smoothing using Numpy vectorized operations.
        - loop over xy_coords as numpy array and calculate smoothed rates in every coordinate pair
        - optimized for speed using numpy operations
        """
        result_list: np.ndarray = np.empty((xy_coords.shape[0], 3), dtype=np.float64)
        for i in range(xy_coords.shape[0]):
            # x_coord = xy_coords[i, 0]
            # y_coord = xy_coords[i, 1]

            result_list[i, :] = [
                xy_coords[i, 0],
                xy_coords[i, 1],
                self.interpolation_function.interpolate(
                    search_window.find_locations_in_radius(
                        xy_coords[i, 0], xy_coords[i, 1]
                    ),
                    self.half_distance_squared,
                ),
            ]

        self.smoothed_data = result_list

    def loop_coords(self) -> None:
        """Perform smoothing using Numpy vectorized operations."""

        search_window = SearchWindow(self.point_data, self.search_radius_squared)
        cell_center = (
            self.grid_settings.grid_size * 0.5
        )  # multiplication faster than division
        y_coords = (
            np.arange(self.grid_settings.rows) * self.grid_settings.grid_size
        ) + cell_center
        x_coords = (
            np.arange(self.grid_settings.cols) * self.grid_settings.grid_size
        ) + cell_center

        # make 2 column arrays of the coordinates x * y grid
        xy_coords = np.column_stack(
            (x_coords.repeat(y_coords.shape[0]), np.tile(y_coords, x_coords.shape[0]))
        )

        # run numpy vectorized smoothing loop
        self.numpy_loop_coords(xy_coords, search_window)
