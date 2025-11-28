# Smoother Module Documentation

A comprehensive spatial data smoothing and interpolation library for Python. This module provides tools for generating, smoothing, and visualizing spatially distributed data using distance-weighted and population-weighted interpolation methods.

**Author:** Toni Patama (tonipat047@gmail.com)  
**Version:** 1.0  
**Date:** November 2025

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Module Components](#module-components)
  - [runner.py](#runnerpy)
  - [smooth_main.py](#smooth_mainpy)
  - [data_generator.py](#data_generatorpy)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Algorithm Details](#algorithm-details)
- [Performance](#performance)
- [Output Formats](#output-formats)

---

## Overview

The Smoother module implements spatial interpolation algorithms for smoothing randomly distributed point data across a regular grid. It's designed for applications such as:

- Population density mapping
- Environmental data interpolation
- Spatial analysis and visualization
- Heatmap generation from sparse point data

The module supports two primary smoothing strategies:
1. **Distance-weighted smoothing**: Based solely on spatial distance
2. **Distance-population weighted smoothing**: Incorporates both distance and population density

---

## Features

✨ **Multiple Interpolation Methods**
- Distance-weighted interpolation with customizable decay functions
- Distance-population weighted interpolation for demographic data
- Extensible `Interpolator` base class for custom methods

🚀 **High Performance**
- Binary search optimization for spatial queries
- Squared distance calculations (avoiding expensive sqrt operations)
- Pre-allocated data structures
- Efficient Y-coordinate sorted data structure

📊 **Flexible Data Generation**
- Random point generation with Gaussian distribution
- Optional population data with configurable ranges
- Customizable grid dimensions and resolutions

💾 **Multiple Output Formats**
- ASCII Grid (.asc) format for GIS applications
- CSV output to stdout
- Matplotlib visualization with customizable plots

🎛️ **Configurable Parameters**
- Grid size and resolution
- Search radius and half-distance decay
- Number of sample points
- Population constraints

---

## Installation

```bash
# Required dependencies
pip install numpy matplotlib

# The module is self-contained
# Simply ensure the smoother/ directory is in your Python path
```

---

## Module Components

### runner.py

**Purpose:** High-level interface for running smoothing operations with pre-configured settings.

**Key Classes:**
- `RunnerSettings`: Dataclass for configuring smoothing operations

**Key Functions:**
- `run(settings: RunnerSettings)`: Execute smoothing with specified configuration

**Smoothing Types:**
- `'distance'`: Distance-weighted interpolation only
- `'distance-population'`: Combined distance and population weighting

**Example Configuration:**
```python
from smoother.runner import RunnerSettings, run

settings = RunnerSettings(
    smooth_func='distance',           # or 'distance-population'
    plot=True,                        # Enable plotting
    half_distance=15_000,             # Decay parameter (meters)
    search_radius=150_000,            # Search radius (meters)
    nrows=1000,                       # Grid rows
    ncols=1000,                       # Grid columns
    grid_size=500,                    # Cell size (meters)
    num_points=100,                   # Number of data points
    max_population=200_000,           # Max population value
    save_smoothed=True,               # Save to .asc file
    print_all=False                   # Print all results or sample
)

run(settings)
```

**RunnerSettings Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smooth_func` | `Literal['distance', 'distance-population']` | `'distance'` | Smoothing algorithm to use |
| `plot` | `bool` | `False` | Generate matplotlib visualization |
| `half_distance` | `int` | `15000` | Distance at which influence is reduced to 50% (meters) |
| `search_radius` | `int` | `150000` | Maximum search radius for neighbors (meters) |
| `nrows` | `int` | `1000` | Number of rows in output grid |
| `ncols` | `int` | `1000` | Number of columns in output grid |
| `grid_size` | `int` | `500` | Grid cell size in meters |
| `num_points` | `int` | `100` | Number of sample points to generate |
| `max_population` | `int` | `200000` | Maximum population value for generation |
| `save_smoothed` | `bool` | `False` | Save output to ASCII grid file |
| `print_all` | `bool` | `False` | Print all results (vs. first 10 rows) |

---

### smooth_main.py

**Purpose:** Core smoothing algorithms and data structures.

#### Key Classes

**`GridMapDefinition`**
```python
@dataclass
class GridMapDefinition:
    rows: int          # Number of grid rows
    cols: int          # Number of grid columns
    grid_size: float   # Size of each grid cell (meters)
```

**`SearchWindow`**
- Finds all data points within a specified radius of a grid cell
- Uses binary search on Y-sorted data for efficiency
- Handles both population and non-population data formats

**Methods:**
- `find_start_index_nb(lower_limit_y: float) -> int`: Binary search for Y-coordinate
- `find_locations_in_radius(current_grid_x, current_grid_y) -> list[tuple]`: Find nearby points

**`Interpolator` (Abstract Base Class)**
- Base class for all interpolation methods
- Subclass to implement custom interpolation algorithms

**`DistanceWeightedInterpolator`**
- Implements inverse distance weighting
- Weight formula: `w = 1 / (1 + (d²/half_distance²))`
- Returns weighted average of nearby point rates

**`DistancePopulationWeightedInterpolator`**
- Extends distance weighting with population density
- Weight formula: `w = population / (1 + (d²/half_distance²))`
- Suitable for demographic data where population density matters

**`Smoother`**
- Main orchestrator class
- Processes entire grid using specified interpolation method

**Methods:**
```python
def __init__(point_data, grid_settings, half_distance, search_radius, interpolation_function)
def smooth()                                  # Perform smoothing
def print(all: bool = False)                  # Print results
def save()                                    # Save to .asc file
def plot(figsize=(12,8), dpi=150)            # Visualize results
```

#### Data Structures

**Point data format (without population):**
```python
(id: int, x: float, y: float, rate: float)
```

**Point data format (with population):**
```python
(id: int, x: float, y: float, population: float, rate: float)
```

**Smoothed output format:**
```python
(x: float, y: float, smoothed_rate: float)
```

---

### data_generator.py

**Purpose:** Generate synthetic spatial data for testing and demonstration.

#### Functions

**`random_point_dist(index, grid_settings, expected_rate, max_population) -> tuple`**
- Generates a random point without population data
- Rate follows Gaussian distribution: `N(μ=expected_rate, σ=expected_rate*0.5)`

**`random_point_dist_pop(index, grid_settings, expected_rate, max_population) -> tuple`**
- Generates a random point with population data
- Population range: `[0.05*max_population, max_population]`
- Rate follows Gaussian distribution

**`create_random_location_data(...) -> list[tuple]`**

**Parameters:**
- `grid_settings: GridMapDefinition` - Grid configuration
- `with_population: bool = False` - Include population data
- `number_of_locations: int = 100` - Number of points to generate
- `expected_rate: float = 100.0` - Mean rate value (μ for Gaussian)
- `max_population: int = 200_000` - Maximum population value

**Returns:**
- List of tuples: `(id, x, y, rate)` or `(id, x, y, population, rate)`

**Distribution Details:**
- **X and Y coordinates**: Uniform distribution across grid extent
- **Rate values**: Gaussian (normal) distribution with `μ=expected_rate`, `σ=0.5*expected_rate`
- **Population values**: Uniform distribution in range `[5%, 100%]` of max_population

---

## Quick Start

### Example 1: Basic Distance Smoothing

```python
from smoother.runner import RunnerSettings, run

# Simple configuration
settings = RunnerSettings(
    smooth_func='distance',
    nrows=500,
    ncols=500,
    grid_size=1000,
    num_points=50,
    half_distance=10_000,
    search_radius=50_000,
    print_all=False
)

run(settings)
```

### Example 2: Population-Weighted Smoothing

```python
from smoother.runner import RunnerSettings, run

settings = RunnerSettings(
    smooth_func='distance-population',
    nrows=1000,
    ncols=1000,
    grid_size=500,
    num_points=100,
    half_distance=15_000,
    search_radius=150_000,
    max_population=500_000,
    save_smoothed=True,
    plot=True
)

run(settings)
```

### Example 3: Using Core Classes Directly

```python
from smoother.smooth_main import (
    GridMapDefinition, 
    Smoother, 
    DistanceWeightedInterpolator
)
from smoother.data_generator import create_random_location_data

# Define grid
grid = GridMapDefinition(rows=500, cols=500, grid_size=1000)

# Generate data
points = create_random_location_data(
    grid, 
    with_population=False,
    number_of_locations=100,
    expected_rate=75.0
)

# Sort by Y coordinate (required for efficiency)
points_sorted = sorted(points, key=lambda p: p[2])

# Create smoother
smoother = Smoother(
    point_data=points_sorted,
    grid_settings=grid,
    half_distance=10_000,
    search_radius=50_000,
    interpolation_function=DistanceWeightedInterpolator()
)

# Execute smoothing
smoother.smooth()

# Output results
smoother.print(all=False)
smoother.plot()
smoother.save()
```

---

## Usage Examples

### Custom Interpolation Method

```python
from smoother.smooth_main import Interpolator

class CustomInterpolator(Interpolator):
    def interpolate(self, window_data, half_distance_squared):
        """Custom interpolation logic"""
        if not window_data:
            return 0.0
        
        # Your custom weighting logic here
        total = 0
        weights = 0
        
        for point in window_data:
            # point format: (id, x, y, rate, distance_sq)
            rate = point[3]
            dist_sq = point[4]
            
            # Example: exponential decay
            weight = math.exp(-dist_sq / half_distance_squared)
            total += weight * rate
            weights += weight
        
        return total / weights if weights > 0 else 0.0

# Use custom interpolator
smoother = Smoother(
    point_data=points_sorted,
    grid_settings=grid,
    half_distance=10_000,
    search_radius=50_000,
    interpolation_function=CustomInterpolator()
)
```

### Generate Data with Different Distributions

```python
from smoother.data_generator import create_random_location_data
from smoother.smooth_main import GridMapDefinition

grid = GridMapDefinition(rows=1000, cols=1000, grid_size=500)

# Low expected rate, sparse data
sparse_data = create_random_location_data(
    grid,
    with_population=True,
    number_of_locations=50,
    expected_rate=25.0,  # Low mean rate
    max_population=100_000
)

# High expected rate, dense data
dense_data = create_random_location_data(
    grid,
    with_population=True,
    number_of_locations=200,
    expected_rate=150.0,  # High mean rate
    max_population=1_000_000
)
```

### Batch Processing Multiple Configurations

```python
from smoother.runner import RunnerSettings, run

configurations = [
    {'smooth_func': 'distance', 'half_distance': 10_000},
    {'smooth_func': 'distance', 'half_distance': 20_000},
    {'smooth_func': 'distance-population', 'half_distance': 15_000},
]

for i, config in enumerate(configurations):
    print(f"\n=== Running configuration {i+1} ===")
    settings = RunnerSettings(**config, save_smoothed=True)
    run(settings)
```

---

## API Reference

### Core Classes

#### `GridMapDefinition`
```python
GridMapDefinition(rows: int, cols: int, grid_size: float)
```
Defines the spatial grid structure.

#### `SearchWindow`
```python
SearchWindow(data_points_sorted: list[list[float]], search_radius_squared: int)
```
Efficiently finds points within radius using binary search.

**Attributes:**
- `data_points_sorted`: Y-sorted list of points
- `search_radius_squared`: Search radius² (optimization)
- `ycoord_list`: Cached Y coordinates for binary search
- `with_population`: Boolean flag for data format

#### `DistanceWeightedInterpolator`
```python
interpolate(window_data, half_distance_squared) -> float
```
Calculates distance-weighted average.

#### `DistancePopulationWeightedInterpolator`
```python
interpolate(window_data, half_distance_squared) -> float
```
Calculates distance and population weighted average.

#### `Smoother`
```python
Smoother(
    point_data: list,
    grid_settings: GridMapDefinition,
    half_distance: int,
    search_radius: int,
    interpolation_function: Interpolator
)
```

**Methods:**
- `smooth()`: Execute smoothing algorithm
- `print(all: bool)`: Output results to stdout
- `save()`: Save as ASCII grid file
- `plot(figsize, dpi)`: Generate matplotlib visualization

---

## Algorithm Details

### Distance-Weighted Interpolation

For each grid cell at position `(x, y)`:

1. **Find neighbors**: All points within `search_radius`
2. **Calculate distances**: `d² = (x - xᵢ)² + (y - yᵢ)²`
3. **Compute weights**: `wᵢ = 1 / (1 + d²/half_distance²)`
4. **Weighted average**: `rate = Σ(wᵢ × rateᵢ) / Σ(wᵢ)`

**Key parameters:**
- `half_distance`: Distance at which weight = 0.5
- `search_radius`: Maximum distance to consider

### Distance-Population Weighted Interpolation

Same as distance weighting, but incorporating population:

1. **Modified weight**: `wᵢ = populationᵢ / (1 + d²/half_distance²)`
2. **Weighted average**: `rate = Σ(wᵢ × rateᵢ) / Σ(wᵢ)`

This gives higher influence to densely populated areas.

### Performance Optimizations

1. **Binary search on Y-coordinates**: O(log n) instead of O(n) for range queries
2. **Squared distances**: Avoids expensive `sqrt()` calculations
3. **Early termination**: Breaks when Y-coordinate exceeds search bounds
4. **Pre-allocated arrays**: Minimizes memory allocations
5. **Cached grid centers**: Pre-computes cell center coordinates

**Time Complexity:**
- Per cell: O(log n + k) where k = neighbors within radius
- Total: O(rows × cols × (log n + k))

**Space Complexity:** O(rows × cols + n)

---

## Performance

### Typical Performance Metrics

**Configuration:** 1000×1000 grid, 100 random points, search_radius=150,000m

- **Pure Python version:** ~6-7 seconds
- **Memory usage:** ~100-200 MB
- **Cells per second:** ~150,000-170,000

**Performance factors:**
- Grid size: Linear impact (rows × cols)
- Search radius: Affects number of neighbors per cell
- Number of points: Logarithmic impact (binary search)
- Half distance: No significant impact (arithmetic only)

### Optimization Tips

1. **Reduce grid resolution** if high precision isn't needed
2. **Decrease search_radius** to limit neighbor searches
3. **Sort data by Y-coordinate** before passing to Smoother (required)
4. **Use appropriate half_distance** - too small causes sharp gradients, too large over-smooths

---

## Output Formats

### Console Output (CSV)

```csv
X,Y,Rate
250,250,45.231
750,250,52.874
1250,250,38.192
...
```

### ASCII Grid File (.asc)

Standard ESRI ASCII grid format:
```
ncols         1000
nrows         1000
xllcorner     0.0
yllcorner     0.0
cellsize      500
NODATA_value  -9999
45.231 52.874 38.192 ...
```

Compatible with:
- QGIS
- ArcGIS
- GDAL/OGR
- Other GIS software

### Matplotlib Visualization

- **Type:** Color mesh plot (pcolormesh)
- **Colorbar:** Shows rate scale
- **Axes:** X and Y coordinates (meters)
- **Customizable:** Figure size and DPI

---

## Examples in Practice

### Urban Planning - Population Density

```python
settings = RunnerSettings(
    smooth_func='distance-population',
    nrows=2000,
    ncols=2000,
    grid_size=250,           # 250m resolution
    num_points=500,          # 500 census blocks
    expected_rate=1000,      # Average population
    max_population=50_000,   # Max per block
    half_distance=5_000,     # 5km influence
    search_radius=20_000,    # 20km search
    save_smoothed=True
)
```

### Environmental Monitoring - Sensor Data

```python
settings = RunnerSettings(
    smooth_func='distance',
    nrows=500,
    ncols=500,
    grid_size=2000,          # 2km resolution
    num_points=50,           # 50 sensors
    expected_rate=25.0,      # Temperature/pollution
    half_distance=10_000,    # 10km influence
    search_radius=50_000,    # 50km search
    save_smoothed=True
)
```

---

## Troubleshooting

**Issue:** Smoothing is too slow
- **Solution:** Reduce grid size, decrease search_radius, or use fewer points

**Issue:** Results are too smooth/blurred
- **Solution:** Decrease `half_distance` or `search_radius`

**Issue:** Results are too sharp/noisy
- **Solution:** Increase `half_distance` or `search_radius`

**Issue:** "Index out of range" error
- **Solution:** Ensure data is sorted by Y-coordinate before passing to Smoother

**Issue:** Memory error
- **Solution:** Reduce grid dimensions (rows × cols)

---

## License

Copyright © 2025 Toni Patama. All rights reserved.

---

## Contact

**Author:** Toni Patama  
**Email:** tonipat047@gmail.com  
**Version:** 1.0  
**Last Updated:** November 2025
