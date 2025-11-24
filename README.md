# Spatial Data Smoothing Module

A high-performance Python implementation for spatial data smoothing and interpolation using distance-weighted methods. This module provides two implementations for performance comparison: a pure Python version and a Numba-optimized version.

## Overview

This module takes randomly distributed point data with associated rates and generates a smoothed grid representation using distance-weighted interpolation. It's designed for applications requiring spatial data analysis, such as:

- Heat map generation
- Spatial rate smoothing
- Geographic data interpolation
- Density estimation

## Implementations

### smoother.py - Pure Python Implementation

The standard Python implementation using object-oriented design with dataclasses and abstract base classes.

**Key Features:**
- Clean, maintainable code structure
- Type hints throughout
- Object-oriented design with `dataclass` decorators
- Abstract base class for extensible interpolation methods
- Comprehensive documentation
- Matplotlib visualization with heatmap plotting
- CSV output to stdout for data export
- Command-line interface with `--all` flag

**Main Components:**
- `LocationData`: Stores point location information (id and coordinates)
- `LocationDataRate`: Associates a rate value with a location
- `SearchWindow`: Finds all data points within a specified radius
- `Interpolator`: Abstract base class for interpolation methods
- `DistanceWeightedInterpolator`: Implements distance-weighted interpolation
- `Smoother`: Main orchestrator for the smoothing process
- `GridMapDefinition`: Defines grid structure (rows, cols, cell size)

**Output Methods:**
- `print()`: Display first 10 results in CSV format
- `print(all=True)`: Display all results in CSV format (X,Y,Rate)
- `plot()`: Generate matplotlib heatmap visualization using pcolormesh

### smoother_njit.py - Numba-Optimized Implementation

High-performance version using Numba's JIT compilation with `@njit` decorators for significant speed improvements.

**Key Features:**
- Numba JIT compilation for near-native performance
- Numpy arrays instead of dataclasses for Numba compatibility
- Normalized smoothing weights for improved accuracy
- Optimized distance calculations and search algorithms
- Same algorithmic approach as pure Python version

**Performance:**
- Typically 10-100x faster than pure Python implementation
- Suitable for large grids and many data points
- First run includes JIT compilation overhead

**Data Structures:**
- LocationData: `[id, x, y]` as numpy array
- LocationDataRate: `[id, x, y, rate]` as numpy array
- LocationFound: `[id, x, y, rate, distance]` as numpy array
- GridCellRate: `[x, y, rate]` as numpy array

## Algorithm

The smoothing process works as follows:

1. **Grid Definition**: Define a grid with specified rows, columns, and cell size
2. **Point Data**: Create or load random point data with associated rates
3. **For Each Grid Cell**:
   - Calculate the cell center coordinates
   - Find all data points within the search radius
   - Apply distance-weighted interpolation using found points
   - Store the smoothed rate value for the grid cell

### Distance-Weighted Interpolation

The interpolation uses an inverse distance weighting formula:

```
weight(d) = 1 / (1 + (d / half_distance)²)
smoothed_rate = Σ(rate_i × weight_i) / Σ(weight_i)
```

Where:
- `d`: Distance from grid cell center to data point
- `half_distance`: Controls the decay rate of distance weighting
- `search_radius`: Maximum distance to search for nearby data points

## Usage

### Pure Python Version

```python
from smoother import Smoother, GridMapDefinition, create_random_location_data

# Define grid
grid_settings = GridMapDefinition(rows=1000, cols=1000, grid_size=500)

# Create random location data
random_locations = create_random_location_data(
    grid_settings, 
    number_of_locations=100
)

# Initialize smoother
my_smoother = Smoother(
    point_data=random_locations,
    grid_settings=grid_settings,
    half_distance=10_000,
    search_radius=150_000
)

# Perform smoothing
my_smoother.smooth()

# Display results (first 10 cells)
my_smoother.print()

# Display all results
my_smoother.print(all=True)

# Visualize with matplotlib
my_smoother.plot()
```

### Command Line Usage

Run the smoother directly from the command line:

```bash
# Run with default output (first 10 cells)
python smoother.py

# Print all smoothed data to stdout (CSV format: X,Y,Rate)
python smoother.py --all

# Save output to file
python smoother.py --all > output.csv
```

The script will prompt you to plot the results after smoothing completes.

### Numba-Optimized Version

```python
from smoother_njit import Smoother, create_random_locations_wrapper

# Grid settings
grid_rows = 1000
grid_cols = 1000
grid_size = 500.0
half_distance = 10000.0
search_radius = 150000.0

# Create random location data
random_locations = create_random_locations_wrapper(
    grid_rows, grid_cols, grid_size, 
    number_of_locations=100
)

# Initialize and run smoother
my_smoother = Smoother(
    random_locations, 
    grid_rows, grid_cols, grid_size,
    half_distance, search_radius
)

my_smoother.smooth()
my_smoother.print_results()
```

## Parameters

- **rows/cols**: Grid dimensions (number of cells)
- **grid_size**: Size of each grid cell (in coordinate units)
- **half_distance**: Controls weight decay rate - smaller values give more weight to nearby points
- **search_radius**: Maximum distance to search for data points - larger values include more points but increase computation time
- **number_of_locations**: Number of random point data to generate

## Performance Comparison

Run both implementations to compare performance:

```bash
# Pure Python version
python smoother.py

# Numba-optimized version
python smoother_njit.py
```

Expected performance characteristics:
- **Pure Python**: Good for small grids (< 100x100) or prototyping ( 15s - 30 s with a grid of 1000*1000 cells))
- **Numba Version**: Recommended for production use and large grids (1000x1000+) takes about 0.4 times less time

## Dependencies

### Pure Python Version (smoother.py)
- Python 3.10+
- matplotlib (for plotting)
- numpy (for grid visualization)

### Numba Version (smoother_njit.py)
- Python 3.10+
- numpy
- numba

## Notes

- The Numba version includes weight normalization in the interpolation function for improved accuracy
- Both implementations use a bounding box optimization before distance calculations
- Generated with assistance from Claude Sonnet 4

