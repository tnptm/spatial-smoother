"""
Optimized Pure Python Smoother - No External Dependencies Beyond Standard Library + matplotlib

This version addresses all performance bottlenecks in pure Python:
1. Inline binary search (no function creation in loop)
2. Pre-extract Y coordinates for faster binary search
3. Use range indexing instead of list slicing
4. Inline distance calculation
5. Cache point references to avoid repeated attribute access
6. Pre-sort data once in Smoother.__init__
"""
import sys
import math
import time
from dataclasses import dataclass
from random import random
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
class GridMapDefinition:
    rows: int
    cols: int
    grid_size: float


class OptimizedSearchWindow:
    """Optimized search window with all bottlenecks removed."""
    
    def __init__(
        self,
        sorted_locations: list[LocationDataRate],
        y_coords: list[float],  # Pre-extracted Y coordinates
        search_radius_squared: float,
    ):
        self.sorted_locations = sorted_locations
        self.y_coords = y_coords  # Faster access than location_data.point[1]
        self.search_radius_squared = search_radius_squared
        self.search_radius = math.sqrt(search_radius_squared)
        self.n_points = len(sorted_locations)

    def find_locations_in_radius(
        self, 
        grid_x: float, 
        grid_y: float
    ) -> list[tuple[LocationDataRate, float]]:
        """Find all locations within search radius using optimized binary search."""
        
        applicable_locations: list[tuple[LocationDataRate, float]] = []
        
        # Cache limits
        lower_y = grid_y - self.search_radius
        upper_y = grid_y + self.search_radius
        
        # Inline binary search to find starting index
        low = 0
        high = self.n_points - 1
        while low <= high:
            mid = (low + high) >> 1  # Bit shift instead of division
            if self.y_coords[mid] < lower_y:
                low = mid + 1
            else:
                high = mid - 1
        start_idx = low
        
        # Use range indexing instead of slicing
        for i in range(start_idx, self.n_points):
            y = self.y_coords[i]
            
            # Early exit when past upper bound
            if y > upper_y:
                break
            
            # Check if within Y bounds (already know it's >= lower_y)
            if y <= upper_y:
                data_point = self.sorted_locations[i]
                x = data_point.location_data.point[0]
                
                # Check X bounding box
                if abs(grid_x - x) <= self.search_radius:
                    # Inline distance calculation (avoid function call)
                    dx = grid_x - x
                    dy = grid_y - y
                    dist_sq = dx * dx + dy * dy
                    
                    if dist_sq <= self.search_radius_squared:
                        applicable_locations.append((data_point, dist_sq))
        
        return applicable_locations


class OptimizedDistanceWeightedInterpolator:
    """Optimized interpolator with minimal overhead."""
    
    __slots__ = ()  # Reduce memory overhead
    
    @staticmethod
    def interpolate(
        window_data: list[tuple[LocationDataRate, float]], 
        half_distance_squared: float
    ) -> float:
        """Interpolate using distance-weighted averaging."""
        if not window_data:
            return 0.0
        
        smoothed_rate = 0.0
        total_weight = 0.0
        
        for data_point, dist_sq in window_data:
            # Inline weight calculation
            weight = 1.0 / (1.0 + dist_sq / half_distance_squared)
            smoothed_rate += weight * data_point.rate
            total_weight += weight
        
        return smoothed_rate / total_weight if total_weight > 0 else 0.0


@dataclass
class GridCellRate:
    point: Point
    rate: float


class OptimizedSmoother:
    """Optimized smoother with all pure Python bottlenecks addressed."""
    
    def __init__(
        self,
        point_data: list[LocationDataRate],
        grid_settings: GridMapDefinition,
        half_distance: float,
        search_radius: float,
    ):
        self.grid_settings = grid_settings
        self.half_distance = half_distance
        self.half_distance_squared = half_distance * half_distance
        self.search_radius = search_radius
        self.search_radius_squared = search_radius * search_radius
        
        # Pre-sort data once (not per SearchWindow)
        print(f"Pre-sorting {len(point_data)} data points...")
        self.sorted_point_data = sorted(
            point_data, 
            key=lambda loc: loc.location_data.point[1]
        )
        
        # Pre-extract Y coordinates for faster binary search
        self.y_coords = [loc.location_data.point[1] for loc in self.sorted_point_data]
        
        self.smoothed_data: list[GridCellRate] = []
        self.interpolator = OptimizedDistanceWeightedInterpolator()

    def smooth(self):
        """Perform optimized smoothing."""
        # Pre-calculate column coordinates
        column_coords = [
            col * self.grid_settings.grid_size
            for col in range(self.grid_settings.cols)
        ]
        cell_half = self.grid_settings.grid_size / 2.0
        
        # Create search window once
        search_window = OptimizedSearchWindow(
            self.sorted_point_data,
            self.y_coords,
            self.search_radius_squared
        )
        
        smoothed_data = []
        total_cells = self.grid_settings.rows * self.grid_settings.cols
        
        print(f"Smoothing {total_cells:,} grid cells...")
        
        # Main smoothing loop
        for row in range(self.grid_settings.rows):
            y_coord = row * self.grid_settings.grid_size
            grid_center_y = y_coord + cell_half
            
            for x_coord in column_coords:
                grid_center_x = x_coord + cell_half
                
                # Find nearby points
                window_data = search_window.find_locations_in_radius(
                    grid_center_x, 
                    grid_center_y
                )
                
                # Interpolate
                smoothed_rate = self.interpolator.interpolate(
                    window_data,
                    self.half_distance_squared
                )
                
                smoothed_data.append(
                    GridCellRate((x_coord, y_coord), smoothed_rate)
                )
        
        self.smoothed_data = smoothed_data

    def print(self, all: bool = False) -> None:
        """Print smoothed data."""
        data_to_print = self.smoothed_data if all else self.smoothed_data[:10]
        print("X,Y,Rate")
        for cell in data_to_print:
            print(f"{cell.point[0]:.2f},{cell.point[1]:.2f},{cell.rate:.2f}")

    def plot(self):
        """Plot the smoothed data using matplotlib."""
        x_coords = [cell.point[0] for cell in self.smoothed_data]
        y_coords = [cell.point[1] for cell in self.smoothed_data]
        rates = [cell.rate for cell in self.smoothed_data]
        
        X = np.array(x_coords).reshape(self.grid_settings.rows, self.grid_settings.cols)
        Y = np.array(y_coords).reshape(self.grid_settings.rows, self.grid_settings.cols)
        Z = np.array(rates).reshape(self.grid_settings.rows, self.grid_settings.cols)
        
        plt.pcolormesh(X, Y, Z, shading='auto')
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Smoothed Data Surface Plot (Optimized)")
        plt.colorbar(label="Rate")
        plt.show()


def create_random_location_data(
    grid_settings: GridMapDefinition, 
    number_of_locations: int
) -> list[LocationDataRate]:
    """Create random test data."""
    rate_data = []
    for loc_id in range(number_of_locations):
        x = random() * grid_settings.cols * grid_settings.grid_size
        y = random() * grid_settings.rows * grid_settings.grid_size
        loc_data = LocationData(loc_id + 1, (x, y))
        rate_data.append(LocationDataRate(loc_data, random() * 100))
    return rate_data


def benchmark_comparison():
    """Compare original vs optimized version."""
    print("=" * 70)
    print("OPTIMIZATION COMPARISON: Original vs Optimized Pure Python")
    print("=" * 70)
    
    # Use moderate grid for comparison
    grid_settings = GridMapDefinition(500, 500, 500)
    half_distance = 10_000
    search_radius = 150_000
    num_points = 100
    
    print(f"\nGrid: {grid_settings.rows}x{grid_settings.cols} = {grid_settings.rows * grid_settings.cols:,} cells")
    print(f"Data points: {num_points}")
    print(f"Search radius: {search_radius:,}")
    
    # Create test data
    random_locations = create_random_location_data(grid_settings, num_points)
    
    # Import original version for comparison
    try:
        from smoother import Smoother as OriginalSmoother
        
        print("\n" + "-" * 70)
        print("ORIGINAL VERSION")
        print("-" * 70)
        
        start = time.time()
        original = OriginalSmoother(
            random_locations,
            grid_settings,
            half_distance,
            search_radius,
        )
        original.smooth()
        original_time = time.time() - start
        
        print(f"Time: {original_time:.4f} seconds")
    except Exception as e:
        print(f"Could not import original: {e}")
        original_time = None
    
    # Test optimized version
    print("\n" + "-" * 70)
    print("OPTIMIZED VERSION")
    print("-" * 70)
    
    start = time.time()
    optimized = OptimizedSmoother(
        random_locations,
        grid_settings,
        half_distance,
        search_radius,
    )
    optimized.smooth()
    optimized_time = time.time() - start
    
    print(f"Time: {optimized_time:.4f} seconds")
    
    # Results
    if original_time:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Original version: {original_time:.4f}s")
        print(f"Optimized version: {optimized_time:.4f}s")
        print(f"\nSpeedup: {original_time / optimized_time:.2f}x faster!")
        print(f"Time saved: {original_time - optimized_time:.4f}s")


if __name__ == "__main__":
    my_smoother_param = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            benchmark_comparison()
            sys.exit(0)
        elif sys.argv[1] == "--all":
            my_smoother_param = True
        else:
            print("\033[91mInvalid argument.\033[0m")
            print("Usage:")
            print("  python3 smoother_optimized_pure.py           # Run with default settings")
            print("  python3 smoother_optimized_pure.py --all     # Print all results")
            print("  python3 smoother_optimized_pure.py --benchmark # Compare with original")
            sys.exit(1)
    
    start_time = time.time()
    grid_settings = GridMapDefinition(1000, 1000, 500)
    half_distance = 10_000
    search_radius = 150_000
    
    random_locations = create_random_location_data(
        grid_settings, 
        number_of_locations=100
    )
    
    my_smoother = OptimizedSmoother(
        random_locations,
        grid_settings,
        half_distance,
        search_radius,
    )
    
    my_smoother.smooth()
    end_time = time.time()
    
    my_smoother.print(all=my_smoother_param)
    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
    
    if not my_smoother_param:
        print("\nDo you want to plot the smoothed data? (y/n)")
        user_input = input().strip().lower()
        if user_input == "y":
            my_smoother.plot()
