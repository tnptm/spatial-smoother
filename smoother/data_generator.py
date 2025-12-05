from random import random, gauss
from .smooth_main import GridMapDefinition


def random_point_dist(
    index: int,
    grid_settings: GridMapDefinition,
    expected_rate: float,
    max_population: int,
) -> tuple:
    """Generate a random data point with population."""
    return (
        index + 1,  # location index
        random() * grid_settings.cols * grid_settings.grid_size,  # X coordinate
        random() * grid_settings.rows * grid_settings.grid_size,  # Y coordinate
        # 0.05 * max_population + random() * max_population* 0.95 , # min population = 5% of max
        gauss(
            mu=expected_rate, sigma=expected_rate * 0.5
        ),  # gaussian distribution, expected values follow the poisson (numpy) distribution
    )


def random_point_dist_pop(
    index: int,
    grid_settings: GridMapDefinition,
    expected_rate: float,
    max_population: int,
) -> tuple:
    """Generate a random data point with population."""
    return (
        index + 1,  # location index
        random() * grid_settings.cols * grid_settings.grid_size,  # X coordinate
        random() * grid_settings.rows * grid_settings.grid_size,  # Y coordinate
        0.05 * max_population
        + random() * max_population * 0.95,  # min population = 5% of max
        gauss(
            mu=expected_rate, sigma=expected_rate * 0.5
        ),  # gaussian distribution, expected values follow the poisson (numpy) distribution
    )


def create_random_location_data(
    grid_settings: GridMapDefinition,
    with_population: bool = False,
    number_of_locations: int = 100,
    expected_rate: float = 100.0,
    max_population: int = 200_000,
) -> list[tuple]:  # list of (index: int, x, y, population(optional), rate)
    """Create random locations within a grid (index, x, y, population, rate).
    - nrows: Number of rows in the grid.
    - xcols: Number of columns in the grid.
    - grid_size: Size of each cell in the grid.
    - number_of_locations: Number of random locations to generate.

    returns: List of tuples (index, x, y, population, rate).
    """
    data_generators = {
        "with_population": random_point_dist_pop,
        "without_population": random_point_dist,
    }

    data_generator = (
        data_generators["with_population"]
        if with_population
        else data_generators["without_population"]
    )

    rate_data = []
    for loc_id in range(number_of_locations):
        loc_data = data_generator(loc_id, grid_settings, expected_rate, max_population)

        rate_data.append(loc_data)

    return rate_data
