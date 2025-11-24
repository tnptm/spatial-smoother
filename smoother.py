import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any  # , Iterable, Tuple

Point = tuple[float, float]


@dataclass
class LocationData:
    id: int
    point: Point


@dataclass
class LocationDataRates:
    location: LocationData
    rate: float


@dataclass
class GridMapDefinition:
    rows: int
    cols: int
    grid_size: float


class SearchWindow:
    random_locations: list[tuple[int, float, float]]
    current_grid_point: tuple[float, float]
    search_radius: int

    def __init__(
        self,
        random_locations: list[tuple[int, float, float]],
        current_grid_point: tuple[float, float],
        search_radius: int,
    ):
        self.random_locations = random_locations
        self.current_grid_point = current_grid_point
        self.search_radius = search_radius

    def calc_distance(self, locationxy: tuple[float, float]) -> float:
        return math.sqrt(
            (self.current_grid_point[0] - locationxy[0]) ** 2
            + (self.current_grid_point[1] - locationxy[1]) ** 2
        )

    def find_locations_in_radius(self):
        # Perform search
        #
        # Before calculating distances get locations in square 2*radius²
        applicable_locations: list[tuple[int, float, float, float]] = []
        for location in self.random_locations:
            if (
                self.current_grid_point[0] - self.search_radius
                <= location[1]
                <= self.current_grid_point[0] + self.search_radius
                and self.current_grid_point[1] - self.search_radius
                <= location[2]
                <= self.current_grid_point[1] + self.search_radius
            ):
                # applicable_locations.append(location)
                distance = self.calc_distance((location[1], location[2]))
                if distance <= self.search_radius:
                    applicable_locations.append(
                        (location[0], location[1], location[2], distance)
                    )
        return applicable_locations


class Interpolator(ABC):
    """Base class for interpolation methods."""

    @abstractmethod
    def interpolate(self, *args: Any, **kwargs: Any) -> float:
        # Perform interpolation
        raise NotImplementedError


class DistanceWeightedInterpolator(Interpolator):
    def __init__(self, window_data: SearchWindow):
        self.window_data: SearchWindow = window_data

    def interpolate(self, x: float, y: float):
        # Perform interpolation
        pass


class PointInterpolator:
    def __init__(
        self,
        window_data: SearchWindow,
        interpolate_function: Interpolator,
    ):
        self.window_data: SearchWindow = window_data
        self.interpolate_function = interpolate_function

    # def interpolate(self, x: float, y: float) -> float:
    # Perform interpolation
    #    pass


class Smoother:
    # data: list[list[float]]
    point_data: list[LocationDataRates]
    smoothed_data: list[list[float]]
    grid_settings: GridMapDefinition
    half_distance: int
    search_radius: int

    def __init__(
        self,
        # data: list[list[float]],
        point_data: list[LocationDataRates],
        grid_settings: GridMapDefinition,
        half_distance: int,
        search_radius: int,
    ):
        # self.data = data
        self.point_data = point_data
        self.grid_settings = grid_settings
        self.half_distance = half_distance
        self.search_radius = search_radius
        # self.smoothed_data = None

    def prepare(self):
        # Prepare data for smoothing, generate rates for each location
        pass

    def smooth(self):
        # Perform smoothing
        # pass
        for row_y in self.data:
            for value in row_y:
                value *= 2

    def print(self):
        # Print smoothed data
        print("not implemented")

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
) -> list[LocationDataRates]:
    """Create random locations within a grid (index,x,y, rate).
    - nrows: Number of rows in the grid.
    - xcols: Number of columns in the grid.
    - grid_size: Size of each cell in the grid.
    - number_of_locations: Number of random locations to generate.

    returns: List of LocationDataRates objects.
    """
    from random import random

    rate_data = []
    for loc_id in range(number_of_locations):
        loc_data = LocationData(
            loc_id + 1,
            random() * grid_settings.cols * grid_settings.grid_size,  # x
            random() * grid_settings.rows * grid_settings.grid_size,  # y
        )
        rate_data.append(LocationDataRates(loc_data, random() * 100))  # max rate = 100

    return rate_data


def create_random_locations(
    xcols: int, nrows: int, grid_size: int, number_of_locations: int
) -> list[LocationDataRates]:
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
        rate_data.append(LocationDataRates(loc_data, random() * 100))  # max rate = 100

    return rate_data


if __name__ == "__main__":
    # data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # create
    # data = create_random_data_matrix(1000, 1000, 0, 100)
    grid_settings = GridMapDefinition(1000, 1000, 500)
    grid_size = grid_settings.grid_size
    half_distance = 10_000
    search_radius = 150_000
    random_locations = create_random_location_data(
        grid_settings, number_of_locations=100
    )

    # point_data = generate_random_rates(random_locations)
    my_smoother = Smoother(
        random_locations, grid_settings, half_distance, search_radius
    )
    # generate rates for each location
    my_smoother.prepare()
    my_smoother.smooth()
    my_smoother.print()
    # my_smoother.save()
    # my_smoother.plot()
    # window_size = 2
    # print(smooth_basic(data, window_size))
