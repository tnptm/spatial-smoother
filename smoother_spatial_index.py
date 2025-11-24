"""
Spatial Index Optimization Example for Smoother

This example demonstrates how to use scipy's KDTree for spatial indexing
to dramatically speed up the neighbor search process.

Performance comparison:
- Without spatial index: O(n*m) - check every point for every grid cell
- With KDTree: O(m*log(n)) - logarithmic search for each grid cell

For 100 points and 1M grid cells:
- Without: 100M operations
- With KDTree: ~6.6M operations (15x fewer!)
"""

import sys
import math
import time
from dataclasses import dataclass
from random import random
from typing import List, Tuple
import numpy as np
from scipy.spatial import KDTree

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
class GridMapDefinition:
    rows: int
    cols: int
    grid_size: float


class SpatialIndexedSmoother:
    """Smoother using KDTree spatial index for fast neighbor searches."""
    
    def __init__(
        self,
        point_data: List[LocationDataRate],
        grid_settings: GridMapDefinition,
        half_distance: float,
        search_radius: float,
    ):
        self.point_data = point_data
        self.grid_settings = grid_settings
        self.half_distance = half_distance
        self.search_radius = search_radius
        self.half_distance_squared = half_distance ** 2
        
        # Build KDTree spatial index from point coordinates
        print("Building KDTree spatial index...")
        build_start = time.time()
        
        self.point_coords = np.array([
            loc.location_data.point for loc in point_data
        ])
        self.point_rates = np.array([
            loc.rate for loc in point_data
        ])
        
        # Build the spatial index
        self.kdtree = KDTree(self.point_coords)
        
        build_time = time.time() - build_start
        print(f"KDTree built in {build_time:.4f} seconds")
        
        self.smoothed_data = []
    
    @staticmethod
    def distance_weight(distance_squared: float, half_distance_squared: float) -> float:
        """Calculate distance weight using squared distances."""
        return 1.0 / (1.0 + distance_squared / half_distance_squared)
    
    def smooth(self):
        """Perform smoothing using KDTree for fast neighbor lookup."""
        cell_half_size = self.grid_settings.grid_size / 2.0
        
        # Pre-allocate results array
        total_cells = self.grid_settings.rows * self.grid_settings.cols
        smoothed_rates = np.zeros(total_cells)
        
        cell_idx = 0
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size
            grid_cell_center_y = y_coord + cell_half_size
            
            for col in range(self.grid_settings.cols):
                x_coord = col * self.grid_settings.grid_size
                grid_cell_center_x = x_coord + cell_half_size
                
                # Use KDTree to find all points within search_radius
                # This is O(log n) instead of O(n)!
                query_point = [grid_cell_center_x, grid_cell_center_y]
                indices = self.kdtree.query_ball_point(query_point, self.search_radius)
                
                if len(indices) > 0:
                    # Get coordinates and rates for nearby points
                    nearby_coords = self.point_coords[indices]
                    nearby_rates = self.point_rates[indices]
                    
                    # Calculate squared distances (vectorized!)
                    dx = nearby_coords[:, 0] - grid_cell_center_x
                    dy = nearby_coords[:, 1] - grid_cell_center_y
                    distances_squared = dx*dx + dy*dy
                    
                    # Calculate weights (vectorized!)
                    weights = 1.0 / (1.0 + distances_squared / self.half_distance_squared)
                    
                    # Calculate weighted average
                    total_weight = np.sum(weights)
                    if total_weight > 0:
                        smoothed_rates[cell_idx] = np.sum(nearby_rates * weights) / total_weight
                
                cell_idx += 1
        
        # Store results
        self.smoothed_data = smoothed_rates.reshape(
            self.grid_settings.rows, 
            self.grid_settings.cols
        )
    
    def print_results(self, sample_size: int = 10):
        """Print sample of results."""
        print("\nX,Y,Rate")
        count = 0
        for row in range(self.grid_settings.rows):
            for col in range(self.grid_settings.cols):
                if count >= sample_size:
                    return
                x = col * self.grid_settings.grid_size
                y = row * self.grid_settings.grid_size
                rate = self.smoothed_data[row, col]
                print(f"{x:.2f},{y:.2f},{rate:.2f}")
                count += 1


class NaiveSmoother:
    """Original smoother without spatial indexing for comparison."""
    
    def __init__(
        self,
        point_data: List[LocationDataRate],
        grid_settings: GridMapDefinition,
        half_distance: float,
        search_radius: float,
    ):
        self.point_data = point_data
        self.grid_settings = grid_settings
        self.half_distance = half_distance
        self.search_radius = search_radius
        self.search_radius_squared = search_radius ** 2
        self.half_distance_squared = half_distance ** 2
        self.smoothed_data = []
        self.operation_count = 0  # Track how many distance checks we do
    
    def smooth(self):
        """Naive O(n*m) smoothing."""
        cell_half_size = self.grid_settings.grid_size / 2.0
        total_cells = self.grid_settings.rows * self.grid_settings.cols
        smoothed_rates = np.zeros(total_cells)
        
        cell_idx = 0
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size
            grid_cell_center_y = y_coord + cell_half_size
            
            for col in range(self.grid_settings.cols):
                x_coord = col * self.grid_settings.grid_size
                grid_cell_center_x = x_coord + cell_half_size
                
                # Check ALL points (no spatial index!)
                total_weight = 0.0
                weighted_sum = 0.0
                
                for data_point in self.point_data:
                    self.operation_count += 1
                    point_x, point_y = data_point.location_data.point
                    
                    # Bounding box check
                    if (abs(grid_cell_center_x - point_x) <= self.search_radius and
                        abs(grid_cell_center_y - point_y) <= self.search_radius):
                        
                        # Calculate squared distance
                        dist_sq = ((grid_cell_center_x - point_x) ** 2 + 
                                  (grid_cell_center_y - point_y) ** 2)
                        
                        if dist_sq <= self.search_radius_squared:
                            weight = 1.0 / (1.0 + dist_sq / self.half_distance_squared)
                            weighted_sum += data_point.rate * weight
                            total_weight += weight
                
                if total_weight > 0:
                    smoothed_rates[cell_idx] = weighted_sum / total_weight
                
                cell_idx += 1
        
        self.smoothed_data = smoothed_rates.reshape(
            self.grid_settings.rows,
            self.grid_settings.cols
        )


def create_random_location_data(
    grid_settings: GridMapDefinition, 
    number_of_locations: int
) -> List[LocationDataRate]:
    """Create random test data."""
    rate_data = []
    for loc_id in range(number_of_locations):
        x = random() * grid_settings.cols * grid_settings.grid_size
        y = random() * grid_settings.rows * grid_settings.grid_size
        loc_data = LocationData(loc_id + 1, (x, y))
        rate_data.append(LocationDataRate(loc_data, random() * 100))
    return rate_data


def benchmark_comparison():
    """Compare naive vs spatial indexed approaches."""
    print("=" * 70)
    print("SPATIAL INDEX PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Use smaller grid for demonstration
    grid_settings = GridMapDefinition(100, 100, 500)  # 10,000 cells
    half_distance = 10_000
    search_radius = 150_000
    num_points = 100
    
    print(f"\nTest parameters:")
    print(f"  Grid: {grid_settings.rows}x{grid_settings.cols} = {grid_settings.rows * grid_settings.cols:,} cells")
    print(f"  Data points: {num_points}")
    print(f"  Search radius: {search_radius:,}")
    print(f"  Expected ops (naive): {grid_settings.rows * grid_settings.cols * num_points:,}")
    
    # Create test data
    random_locations = create_random_location_data(grid_settings, num_points)
    
    # Test 1: Naive approach
    print("\n" + "-" * 70)
    print("NAIVE APPROACH (no spatial index)")
    print("-" * 70)
    naive_smoother = NaiveSmoother(
        random_locations,
        grid_settings,
        half_distance,
        search_radius,
    )
    
    start = time.time()
    naive_smoother.smooth()
    naive_time = time.time() - start
    
    print(f"Time: {naive_time:.4f} seconds")
    print(f"Operations: {naive_smoother.operation_count:,}")
    
    # Test 2: KDTree approach
    print("\n" + "-" * 70)
    print("KDTREE SPATIAL INDEX")
    print("-" * 70)
    kdtree_smoother = SpatialIndexedSmoother(
        random_locations,
        grid_settings,
        half_distance,
        search_radius,
    )
    
    start = time.time()
    kdtree_smoother.smooth()
    kdtree_time = time.time() - start
    
    print(f"Time: {kdtree_time:.4f} seconds")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Naive approach:  {naive_time:.4f}s")
    print(f"KDTree approach: {kdtree_time:.4f}s")
    print(f"\nSpeedup: {naive_time / kdtree_time:.2f}x faster with spatial index!")
    print(f"Time saved: {naive_time - kdtree_time:.4f}s ({(1 - kdtree_time/naive_time)*100:.1f}% reduction)")
    
    # Verify results are similar
    diff = np.abs(naive_smoother.smoothed_data - kdtree_smoother.smoothed_data)
    print(f"\nMax difference in results: {np.max(diff):.6f} (should be ~0)")
    print(f"Mean difference: {np.mean(diff):.6f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_comparison()
