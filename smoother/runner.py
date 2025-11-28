#!/usr/bin/env python3
"""
Docstring for smoother.runner
This module contains the runner for the smoother module, allowing users to run different smoothing functions
with specified settings such as grid size, number of points, and smoothing parameters.
It also includes a dataclass for encapsulating runner settings.
author: Toni Patama tonipat047@gmail.com, 2025-11-28, version 1.0

"""

from .smooth_main import GridMapDefinition, DistanceWeightedInterpolator, DistancePopulationWeightedInterpolator, Smoother
from .data_generator import create_random_location_data
from dataclasses import dataclass
from typing import Literal
import time


SMOOTHER_TYPES = ['distance', 'distance-population']

SMOOTHERS = {
    'distance': DistanceWeightedInterpolator,
    'distance-population': DistancePopulationWeightedInterpolator,
}

@dataclass
class RunnerSettings:
    """Settings for the smoother runner.
    
    Configures how the smoothing process is performed and optionally visualized.
    Parameters:
        smooth_func (Literal['distance', 'distance-population'], default='distance'):
            The smoothing strategy to use.
            - 'distance': Smooth values based solely on spatial distance.
            - 'distance-population': Incorporates both spatial distance and nearby population
              density into the smoothing.
        plot (bool, default=False):
            If True, generates visual plots of the smoothing results for inspection or debugging.
        half_distance (int, default=15000):
            The half-distance parameter (in meters) used by the smoothing kernel or decay function.
            Typically represents the distance at which influence is reduced to 50%.
        search_radius (int, default=150000):
            Maximum search radius (in meters) for neighbors considered during smoothing.
            Points beyond this radius are ignored.
        nrows (int, default=1000):
            Number of rows in the output grid or raster used for smoothing computations.
        ncols (int, default=1000):
            Number of columns in the output grid or raster used for smoothing computations.
        grid_size (int, default=500):
            The grid cell size (in meters). Determines spatial resolution: smaller values
            produce finer grids with more cells.
        num_points (int, default=100):
            Number of sample or synthetic points to use in generating or testing the smoothing.
            Depending on the pipeline, may represent input points per tile or batch.
        max_population (int, default=200000):
            Upper cap for population values considered in 'distance-population' mode to avoid
            extreme influence from highly populated areas.
        save_smoothed (bool, default=False):
            If True, saves the smoothed output (e.g., grid or raster) to disk.
        print_all (bool, default=False):
            If True, enables verbose logging for debugging, printing intermediate steps
            and statistics throughout smoothing.
    
    """
    smooth_func: Literal['distance', 'distance-population'] = 'distance'  # default can be 'distance'
    plot: bool = False
    half_distance: int = 15_000
    search_radius: int = 150_000
    nrows: int = 1000
    ncols: int = 1000
    grid_size: int = 500
    num_points: int = 100
    max_population: int = 200_000
    save_smoothed: bool = False
    print_all: bool = False


    
def run(settings: RunnerSettings):
    """Run specified smoothing function. Run distance weighting smoother and time it."""

    start_time = time.time()
    grid_settings = GridMapDefinition(settings.nrows, settings.ncols, settings.grid_size)
    # grid_size = grid_settings.grid_size
    #half_distance = settings.half_distance
    #search_radius = settings.search_radius

    print(f"Running '{settings.smooth_func.capitalize()}'-weighting smoother...")

    print(f"Generating {settings.num_points} random location datasets...")
    random_locations = create_random_location_data(
        grid_settings, 
        with_population=True if settings.smooth_func == 'distance-population' else False,
        number_of_locations=settings.num_points,
        expected_rate=settings.num_points,
        max_population=settings.max_population if hasattr(settings, 'max_population') else 200_000
    )

    # sort random locations by Y coordinate for efficient searching
    random_locations = sorted(
        random_locations,
        key=lambda loc: loc[2],  # sort by Y coordinate
    )

    # point_data = generate_random_rates(random_locations)
    my_smoother = Smoother(
        random_locations,
        grid_settings,
        settings.half_distance,
        settings.search_radius,
        interpolation_function=SMOOTHERS[settings.smooth_func]() # load appropriate smoother object
    )
    # generate rates for each location
    # my_smoother.prepare()
    print("Smoothing data...")
    my_smoother.smooth()

    if settings.save_smoothed:
        print("Saving smoothed data to file...")
        my_smoother.save()

    print("Smoothing completed.")
    end_time = time.time()

    my_smoother.print(all=settings.print_all if hasattr(settings, 'print_all') else False)

    print(f"\nTime elapsed: {end_time - start_time} seconds")
    # my_smoother.save() TODO

    if settings.plot:
        print("Do you want to plot the smoothed data? (y/n)")
        user_input = input().strip().lower()
        if user_input == "y":
            my_smoother.plot()
        else:
            print("Plotting skipped.")

if __name__ == "__main__":
    # sample run
    settings = RunnerSettings(
        smooth_func='distance-population',
        plot=True,
        half_distance=15_000,
        search_radius=150_000,
        nrows=1000,
        ncols=1000,
        grid_size=500,
        num_points=100,
        max_population=200_000,
        save_smoothed=True,
        print_all=False
    )
    run(settings)