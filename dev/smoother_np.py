"""
Random Data Smoothing Application

This module provides a framework for spatial data smoothing and interpolation using
distance-weighted methods. It's designed to take randomly distributed point data
with associated rates and generate a smoothed grid representation.

Main Components:
---------------
- LocationData: Stores point location information (id and coordinates)
- LocationDataRate: Associates a rate value with a location
- SearchWindow: Finds all data points within a specified radius of a grid point
- Interpolator: Abstract base class for interpolation methods
- DistanceWeightedInterpolator: Implements distance-weighted interpolation
- Smoother: Main class that orchestrates the smoothing process across a grid
- GridMapDefinition: Defines the grid structure (rows, cols, cell size)

Usage:
------
1. Define a grid using GridMapDefinition(rows, cols, grid_size)
2. Create or load point data with associated rates (LocationDataRate objects)
3. Initialize a Smoother with the grid settings, data, and interpolation parameters
4. Call smooth() to generate interpolated values for each grid cell
5. Use print(), save(), or plot() methods to output results

The smoothing process:
---------------------
For each grid cell center:
  1. Find all data points within the search radius
  2. Calculate distance-weighted interpolation using those points
  3. Store the smoothed rate value for that grid cell

Parameters:
----------
- half_distance: Controls the decay rate of distance weighting
- search_radius: Maximum distance to search for nearby data points
"""
import sys
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import random
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

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
        random_locations: List of random locations.
        current_grid_point: Current grid point.
        search_radius: Search radius.

    Returns:
        List of found locations within the search radius.
    """

    random_locations: list[LocationDataRate]
    search_radius_squared: int

    def __init__(
        self,
        random_locations: list[LocationDataRate],
        search_radius_squared: int,
    ):
        self.random_locations = random_locations
        self.search_radius_squared = search_radius_squared
        self.search_radius = math.sqrt(search_radius_squared)
        #self.data_points_sorted = sorted(
        #    self.random_locations, key=lambda loc: loc.location_data.point[1] # sort by Y coordinate
        #)
        self.np_data_points = np.array(
            [[id, *loc.location_data.point, loc.rate]  for id,loc in enumerate(self.random_locations)]
        )

    @staticmethod
    def calc_distance_squared(grid_p_x: float, grid_p_y: float, point_x: float, point_y: float) -> float:
        """Calculate squared distance between two points."""
        return (grid_p_x - point_x) ** 2 + (grid_p_y - point_y) ** 2
        

    def find_locations_in_radius(self, current_grid_x: float, current_grid_y: float) -> np.ndarray:
        # Perform search
        #
        # Before calculating distances get locations in square 2*radius²
        #applicable_locations: list[tuple[LocationDataRate, float]] = []
        #found_y = False
        
        # caching limits
        #limit_y = current_grid_y + self.search_radius
        #lower_limit_y = current_grid_y - self.search_radius
        
        #limit_x = current_grid_x + self.search_radius        
        grid_point = np.array([current_grid_x, current_grid_y])
        dists = np.linalg.norm(self.np_data_points[:, 1:3] - grid_point, axis=1)
        # add dists as a new column into the numby array self.np_data_points
        np_distance_data = np.column_stack((self.np_data_points[dists <= self.search_radius], dists[dists <= self.search_radius]**2))
        #selected_points = np_distance_data[dists <= self.search_radius]
        #selected_points = self.np_data_points[dists <= self.search_radius]
        #for point in selected_points:
        #    loc_data_rate = LocationDataRate(
        #        LocationData(int(point[0])+1, (point[1], point[2])),
        #        point[3],
        #    )
        #    distance_squared = point[4]
        #    applicable_locations.append( (loc_data_rate, distance_squared) )
        return np_distance_data #applicable_locations


class Interpolator(ABC):
    """Base class for interpolation methods."""

    @abstractmethod
    def interpolate(self, *args: Any, **kwargs: Any) -> float:
        # Perform interpolation
        raise NotImplementedError


class DistanceWeightedInterpolator(Interpolator):
    #def __init__(self)#, window_data: list[LocationFound]):
    #    self.window_data: list[LocationFound] = window_data

    @staticmethod
    def distance_weight_sq(distance_squared: float, half_distance_squared: float) -> float:
        """Calculate distance weight based on squared distance."""
        # Optional alternative method using distance instead of squared distance
        # @staticmethod
        # def distance_weight(self, distance: float, half_distance: float) -> float:
        #     return 1 / (1 + (distance / half_distance)**2 )
        return 1 / (1 + (distance_squared / half_distance_squared) )
    
    
    def interpolate(self, window_data: np.ndarray, half_distance_squared: float) -> float:
        """Interpolate the rate based on distance-weighted averaging.

        Args:
            half_distance_squared: The half-distance squared used for weighting.
            window_data: np.ndarray with columns [id, x, y, rate, distance_squared]

        Returns:
            The interpolated rate.
        """
        smoothed_rate = 0
        total_weights = 0
        for point_data in window_data:
            # Perform interpolation using point_data
            dist_weight = self.distance_weight_sq(
                point_data[4], half_distance_squared    
            )
            smoothed_rate += dist_weight * point_data[3]
            total_weights += dist_weight

        if total_weights == 0:
            return 0
        weighted_mean = smoothed_rate / total_weights
        
        return weighted_mean


@dataclass
class GridCellRate:
    point: Point
    rate: float


class Smoother:
    """Smoother class for smoothing data using search window data."""

    # data: list[list[float]]
    point_data: list[LocationDataRate]
    smoothed_data: list[GridCellRate]
    grid_settings: GridMapDefinition
    half_distance: int
    search_radius: int

    def __init__(
        self,
        # data: list[list[float]],
        point_data: list[LocationDataRate],
        grid_settings: GridMapDefinition,
        half_distance: int,
        search_radius: int,
        # interpolation_function: Interpolator,
    ):
        # self.data = data
        self.point_data = point_data
        self.grid_settings = grid_settings
        self.half_distance = half_distance
        self.half_distance_squared = half_distance ** 2
        self.search_radius = search_radius
        # self.interpolation_function = interpolation_function
        self.smoothed_data = []
        self.search_radius_squared = search_radius ** 2

    def prepare(self):
        # Prepare data for smoothing, generate rates for each location
        pass

    def smooth(self):
        """
        Perform smoothing using search_window_data

        It uses a search window to find nearby data points and calculates the average value within the window.

        """
        # Perform smoothing
        # pass
        ## CREATE colID,Xcoord list
        column_coord_list = [
            column_id * self.grid_settings.grid_size
            for column_id in range(self.grid_settings.cols)
        ]
        cell_center = self.grid_settings.grid_size / 2

        search_window = SearchWindow(
            self.point_data, self.search_radius_squared
        )
        interpolator: DistanceWeightedInterpolator = DistanceWeightedInterpolator()

        smoothed_data = []
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size
            for x_coord in column_coord_list:
                # define search window data
                grid_cell_center_x = x_coord + cell_center
                grid_cell_center_y = y_coord + cell_center

                #interpolator = DistanceWeightedInterpolator(search_window.find_locations_in_radius(grid_cell_center_x, grid_cell_center_y))
                smoothed_rate_xy = interpolator.interpolate(
                    search_window.find_locations_in_radius(grid_cell_center_x, grid_cell_center_y),
                    self.half_distance_squared)

                smoothed_data.append(
                    GridCellRate(Point((x_coord, y_coord)), smoothed_rate_xy)
                )

        self.smoothed_data = smoothed_data

    def print(self, all:bool=False) -> None:
        """Print smoothed data of 10 first cells as a sample"""
        data_to_print = self.smoothed_data if all else self.smoothed_data[:10]
        print("X,Y,Rate")
        for cell in data_to_print:
            # for cell in row:
            print(
                #f"X: {cell.point[0]:.2f} Y: {cell.point[1]:.2f} Rate: {cell.rate:.2f}",
                #end=" ",
                f"{cell.point[0]:.2f},{cell.point[1]:.2f},{cell.rate:.2f}",
            )
            # print()

    def save(self):
        # Save smoothed data to file
        pass

    def plot(self):
        """Plot the smoothed data using matplotlib and make a mesh plot.
        Args:
            None
        """
        
        x_coords = [cell.point[0] for cell in self.smoothed_data]
        y_coords = [cell.point[1] for cell in self.smoothed_data]
        rates = [cell.rate for cell in self.smoothed_data]

        # Reshape data for grid plotting
        X = np.array(x_coords).reshape(self.grid_settings.rows, self.grid_settings.cols)
        Y = np.array(y_coords).reshape(self.grid_settings.rows, self.grid_settings.cols)
        Z = np.array(rates).reshape(self.grid_settings.rows, self.grid_settings.cols)

        plt.pcolormesh(X, Y, Z, shading='auto')
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Smoothed Data Surface Plot")
        plt.colorbar(label="Rate")
        plt.show()


#def create_grid_map(
#    rows: int, cols: int, min_val: float, max_val: float
#) -> list[list[float]]:
#    """Create a random data matrix."""
    #import random

    #return [
    #    [random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)
    #]


def create_random_location_data(
    grid_settings: GridMapDefinition, number_of_locations: int
) -> list[LocationDataRate]:
    """Create random locations within a grid (index,x,y, rate).
    - nrows: Number of rows in the grid.
    - xcols: Number of columns in the grid.
    - grid_size: Size of each cell in the grid.
    - number_of_locations: Number of random locations to generate.

    returns: List of LocationDataRates objects.
    """

    rate_data = []
    for loc_id in range(number_of_locations):
        loc_data = LocationData(
            loc_id + 1,
            Point(
                [
                    random() * grid_settings.cols * grid_settings.grid_size,
                    random() * grid_settings.rows * grid_settings.grid_size,
                ]
            ),  # Point(x,y)
        )
        rate_data.append(LocationDataRate(loc_data, random() * 100))  # max rate = 100

    return rate_data


def create_random_locations(
    xcols: int, nrows: int, grid_size: int, number_of_locations: int
) -> list[LocationDataRate]:
    """Create random locations within a grid (index,x,y, rate).
    - nrows: Number of rows in the grid.
    - xcols: Number of columns in the grid.
    - grid_size: Size of each cell in the grid.
    - number_of_locations: Number of random locations to generate.
    """
    from random import random

    rate_data = []
    for loc_id in range(number_of_locations):
        loc_data = LocationData(
            loc_id + 1,
            random() * xcols * grid_size,  # x
            random() * nrows * grid_size,  # y
        )
        rate_data.append(LocationDataRate(loc_data, random() * 100))  # max rate = 100

    return rate_data


if __name__ == "__main__":
    # Measeure time in ms
    my_smoother_param = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            my_smoother_param = True
        else:
            print("\033[91mInvalid argument.\033[0m")
            print("    Use '--all' to print all smoothed data to stdout, to view or save results.\n")
            print("\033[92m    Usage: python3 smoother.py [--all]\033[0m\n")
            sys.exit(1)
    #else:
    #    my_smoother.print()
    start_time = time.time()
    grid_settings = GridMapDefinition(1000, 1000, 500)
    grid_size = grid_settings.grid_size
    half_distance = 10_000
    search_radius = 150_000
    random_locations = create_random_location_data(
        grid_settings, number_of_locations=100
    )

    # point_data = generate_random_rates(random_locations)
    my_smoother = Smoother(
        random_locations,
        grid_settings,
        half_distance,
        search_radius,
    )
    # generate rates for each location
    # my_smoother.prepare()
    my_smoother.smooth()
    end_time = time.time()

    my_smoother.print(all=my_smoother_param)

    print(f"time taken: {end_time - start_time} seconds")
    # my_smoother.save() TODO
    
    print("Do you want to plot the smoothed data? (y/n)")
    user_input = input().strip().lower()
    if user_input == "y":
        my_smoother.plot()
    else:
        print("Plotting skipped.")
