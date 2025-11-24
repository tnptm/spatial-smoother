import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import random

# from pickletools import stackslice
from typing import Any  # , Iterable, Tuple

# from numba import njit

Point = tuple[float, float]
# @dataclass
# class GridCell:
#    row: int
#    col: int


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
    random_locations: list[LocationDataRate]
    current_grid_point: Point
    search_radius: int

    def __init__(
        self,
        random_locations: list[LocationDataRate],
        current_grid_point: Point,
        search_radius: int,
    ):
        self.random_locations = random_locations
        self.current_grid_point = current_grid_point
        self.search_radius = search_radius

    @staticmethod
    def calc_distance(grid_xy: Point, locationxy: Point) -> float:
        return math.sqrt(
            (grid_xy[0] - locationxy[0]) ** 2 + (grid_xy[1] - locationxy[1]) ** 2
        )

    def find_locations_in_radius(self) -> list[LocationFound]:
        # Perform search
        #
        # Before calculating distances get locations in square 2*radius²
        applicable_locations: list[LocationFound] = []
        for data_point in self.random_locations:
            if (
                abs(self.current_grid_point[0] - data_point.location_data.point[0])
                <= self.search_radius
                and abs(self.current_grid_point[1] - data_point.location_data.point[1])
                <= self.search_radius
            ):
                # applicable_locations.append(location)
                distance = self.calc_distance(
                    self.current_grid_point, data_point.location_data.point
                )
                if distance <= self.search_radius:
                    applicable_locations.append(LocationFound(data_point, distance))
        return applicable_locations


class Interpolator(ABC):
    """Base class for interpolation methods."""

    @abstractmethod
    def interpolate(self, *args: Any, **kwargs: Any) -> float:
        # Perform interpolation
        raise NotImplementedError


class DistanceWeightedInterpolator(Interpolator):
    def __init__(self, window_data: list[LocationFound]):
        self.window_data: list[LocationFound] = window_data

    @staticmethod
    def distance_weight(distance: float, half_distance: float) -> float:
        return 1 / (1 + (distance / half_distance) ** 2)

    def interpolate(self, half_distance: float) -> float:
        # Perform interpolation
        # pass

        smoothed_rate = 0
        sum_weights = 0
        for point_data in self.window_data:
            # Perform interpolation using point_data
            weight = point_data.point_data.rate * self.distance_weight(
                point_data.distance, half_distance
            )
            weighted_rate = weight * point_data.point_data.rate
            smoothed_rate += weighted_rate
            sum_weights += weight

        if sum_weights == 0:
            return 0
        weighted_mean = smoothed_rate / sum_weights
        return weighted_mean


# class PointInterpolator:
#    def __init__(
#        self,
#        window_data: SearchWindow,
#        interpolate_function: Interpolator,
#    ):
#        self.window_data: SearchWindow = window_data
#        self.interpolate_function = interpolate_function

# def interpolate(self, x: float, y: float) -> float:
# Perform interpolation
#    pass


@dataclass
class GridCellRate:
    point: Point
    rate: float


class Smoother:
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
        self.search_radius = search_radius
        # self.interpolation_function = interpolation_function
        self.smoothed_data = []

    def prepare(self):
        # Prepare data for smoothing, generate rates for each location
        pass

    def smooth(self):
        # Perform smoothing
        # pass
        ## CREATE colID,Xcoord list
        column_coord_list = [
            column_id * self.grid_settings.grid_size
            for column_id in range(self.grid_settings.cols)
        ]

        smoothed_data = []
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size
            for x_coord in column_coord_list:
                # value = 0
                # count = 0
                # define search window data
                grid_cell_center_x = x_coord + (self.grid_settings.grid_size / 2)
                grid_cell_center_y = y_coord + self.grid_settings.grid_size / 2
                grid_cell_center = Point([grid_cell_center_x, grid_cell_center_y])
                search_window_points = SearchWindow(
                    self.point_data, grid_cell_center, self.search_radius
                ).find_locations_in_radius()

                interpolator = DistanceWeightedInterpolator(search_window_points)
                smoothed_rate_xy = interpolator.interpolate(self.half_distance)

                smoothed_data.append(
                    GridCellRate(Point((x_coord, y_coord)), smoothed_rate_xy)
                )

        self.smoothed_data = smoothed_data
        # search_window_data = [
        #    point_data
        #    for point_data in self.point_data
        #    if  <= self.search_radius
        #    #and abs(point_data.y - y_coord) <= self.search_radius
        # ]
        # Perform smoothing using search_window_data

    def print(self) -> None:
        # Print smoothed data
        # print("not implemented")
        # print(self.smoothed_data[0])
        for cell in self.smoothed_data[:10]:
            # for cell in row:
            print(
                f"X: {cell.point[0]:.2f} Y: {cell.point[1]:.2f} Rate: {cell.rate:.2f}",
                end=" ",
            )
            # print()

    def save(self):
        # Save smoothed data to file
        pass

    def plot(self):
        # Plot smoothed data
        pass


def create_grid_map(
    rows: int, cols: int, min_val: float, max_val: float
) -> list[list[float]]:
    """Create a random data matrix."""
    import random

    return [
        [random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)
    ]


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
    # data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # create
    # data = create_random_data_matrix(1000, 1000, 0, 100)
    #
    #
    # Measeure time in ms
    start_time = time.time()
    grid_settings = GridMapDefinition(1000, 1000, 500)
    grid_size = grid_settings.grid_size
    half_distance = 10_000
    search_radius = 150_000
    random_locations = create_random_location_data(
        grid_settings, number_of_locations=100
    )

    # define interpolator object
    #    interpolator = DistanceWeightedInterpolator()

    # point_data = generate_random_rates(random_locations)
    my_smoother = Smoother(
        random_locations,
        grid_settings,
        half_distance,
        search_radius,
        #        interpolation_function=linear_interpolation,
    )
    # generate rates for each location
    # my_smoother.prepare()
    my_smoother.smooth()
    end_time = time.time()
    my_smoother.print()
    print(f"time taken: {end_time - start_time} seconds")
    # my_smoother.save()
    # my_smoother.plot()
    # window_size = 2
    # print(smooth_basic(data, window_size))
