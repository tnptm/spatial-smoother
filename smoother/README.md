# Smoother Module Documentation

A comprehensive spatial data smoothing and interpolation library for Python. This module provides tools for generating, smoothing, and visualizing spatially distributed data using distance-weighted and population-weighted interpolation methods.

**Author:** Toni Patama (tonipat047@gmail.com)  
**Version:** 1.1  
**Date:** December 2025

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Performance Comparison](#performance-comparison)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)

---

## Overview

The Smoother module implements spatial interpolation algorithms for smoothing randomly distributed point data across a regular grid. It is designed for high performance and flexibility, offering multiple implementation strategies ("loopers") to balance speed and dependency requirements.

Key applications include:
- Population density mapping
- Environmental data interpolation
- Spatial analysis and visualization

---

## Project Structure

The project is organized into a modular structure to separate concerns and allow for interchangeable implementations.

### Root Directory (`app/`)
- **`main.py`**: The main entry point and demonstration script. It runs performance comparisons between different smoothing implementations (Pure Python, Numpy, Numba) and demonstrates different weighting strategies.

### Smoother Package (`app/smoother/`)
- **`runner.py`**: High-level interface for running smoothing operations. Defines `RunnerSettings` to configure parameters like grid size, search radius, and the chosen "looper" implementation.
- **`smooth_main.py`**: Contains the core abstract base classes (`Smoother`, `InterpolateLoop`, `Interpolator`) and standard implementations:
    - `PurePythonLooper`: Reference implementation using standard Python lists and loops.
    - `NumpyVectorizedLooper`: Optimized implementation using Numpy vectorization.
    - `DistanceWeightedInterpolator`: Standard inverse-distance weighting.
    - `DistancePopulationWeightedInterpolator`: Population-aware weighting.
- **`numba_jit_looper.py`**: High-performance implementation using Numba JIT compilation (`@njit`).
    - `NumbaJitLooper`: Uses compiled machine code for the heavy lifting (neighbor search and interpolation).
- **`data_generator.py`**: Utilities for generating synthetic spatial data (random points with Gaussian-distributed rates) for testing and benchmarking.

---

## Features

✨ **Multiple Implementation Strategies ("Loopers")**
- **Pure Python**: No extra dependencies (other than standard libs), good for debugging and reference.
- **Numpy Vectorized**: Uses array operations for better performance on large datasets.
- **Numba JIT**: Just-In-Time compilation for near-C speeds (fastest).

✨ **Flexible Interpolation**
- **Distance-weighted**: Standard inverse distance weighting ($1 / (1 + d^2/h^2)$).
- **Distance-population weighted**: Incorporates population density into the weighting.

✨ **High Performance**
- Binary search optimization for spatial queries (in all implementations).
- Squared distance calculations to avoid expensive square roots.
- Efficient memory management with pre-allocated arrays.

---

## Performance Comparison

Based on benchmarks running `main.py` (1000x1000 grid, 100 points):

| Implementation | Description | Relative Speed | Notes |
|----------------|-------------|----------------|-------|
| **Pure Python** | `looper_func='pure-python-looper'` | Baseline | Reference implementation. ~7.5s |
| **Numpy** | `looper_func='numpy-vectorized'` | Slower | Overhead of array creation for small batches can make it slower than pure Python for this specific algorithm structure. |
| **Numba JIT** | `looper_func='numba-jit'` | **Fastest** | **~2x speedup** (3.5s vs 7.5s). Compiles critical loops to machine code. |

*Note: Performance varies based on grid size, number of points, and hardware.*

---

## Installation

Dependencies are managed via `requirements.txt` or `pyproject.toml`.

```bash
# Core dependencies
pip install numpy matplotlib

# Optional (for high performance)
pip install numba
```

---

## Usage

### Running the Demo
To see the comparisons in action, run the main script from the `app` directory:

```bash
python main.py
```

### Using the Library
You can use the `runner` module to configure and run smoothing tasks programmatically.

```python
from smoother.runner import RunnerSettings, run

# Configure settings
settings = RunnerSettings(
    smooth_func='distance',           # 'distance' or 'distance-population'
    looper_func='numba-jit',          # 'pure-python-looper', 'numpy-vectorized', or 'numba-jit'
    plot=True,                        # Enable plotting
    half_distance=10_000,             # Decay parameter (meters)
    search_radius=150_000,            # Search radius (meters)
    nrows=1000,                       # Grid rows
    ncols=1000,                       # Grid columns
    grid_size=500,                    # Cell size (meters)
    num_points=100,                   # Number of data points
    save_smoothed=True,                # Save to .asc file
    print_all=False,
    looper_funcLiteral["numpy-vectorized", "pure-python-looper", "numba-jit"] = (
        "pure-python-looper"
    )
)

# Run the smoother
run(settings)
```

---

## API Reference

### `RunnerSettings`
Configuration dataclass in `runner.py`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smooth_func` | `str` | `'distance'` | Interpolation strategy. |
| `looper_func` | `str` | `'pure-python-looper'` | Implementation backend (`pure-python-looper`, `numpy-vectorized`, `numba-jit`). |
| `half_distance` | `int` | `15000` | Distance decay parameter. |
| `search_radius` | `int` | `150000` | Max search radius. |
| `nrows`, `ncols` | `int` | `1000` | Grid dimensions. |
| `grid_size` | `int` | `500` | Cell size in meters. |

### `Smoother` (Base Class)
Located in `smooth_main.py`. Orchestrates the smoothing process.
- `smooth()`: Executes the smoothing logic using the configured looper.
- `save()`: Exports results to ASCII grid format.
- `plot()`: Visualizes the result using Matplotlib.

---

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

Copyright (c) 2025 Toni Patama.
