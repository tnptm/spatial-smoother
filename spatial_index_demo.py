"""
Simple demonstration of spatial indexing benefits.

Key Concepts:
1. Without spatial index: Check EVERY point for EVERY grid cell (O(n*m))
2. With spatial index (KDTree): Only check nearby points (O(m*log(n)))

The bigger the difference between grid cells (m) and data points (n),
the more dramatic the speedup!
"""

import time
import numpy as np
from scipy.spatial import KDTree


def demo_basic_usage():
    """Show how to use KDTree for spatial queries."""
    print("=" * 70)
    print("SPATIAL INDEX BASICS: How KDTree Works")
    print("=" * 70)
    
    # Create some random 2D points
    np.random.seed(42)
    points = np.random.rand(10, 2) * 1000  # 10 points in 1000x1000 space
    
    print("\n10 Random Points:")
    for i, (x, y) in enumerate(points):
        print(f"  Point {i}: ({x:.1f}, {y:.1f})")
    
    # Build KDTree
    print("\nBuilding KDTree...")
    kdtree = KDTree(points)
    
    # Query: Find all points within radius 300 of location (500, 500)
    query_point = [500, 500]
    radius = 300
    
    print(f"\nQuery: Find all points within {radius} units of {query_point}")
    
    # Method 1: Using KDTree (fast!)
    start = time.perf_counter()
    indices = kdtree.query_ball_point(query_point, radius)
    kdtree_time = time.perf_counter() - start
    
    print(f"\nKDTree result: Found {len(indices)} points")
    for idx in indices:
        x, y = points[idx]
        dist = np.sqrt((x - query_point[0])**2 + (y - query_point[1])**2)
        print(f"  Point {idx}: ({x:.1f}, {y:.1f}) - distance: {dist:.1f}")
    print(f"  Query time: {kdtree_time*1e6:.2f} microseconds")
    
    # Method 2: Linear search (slow!)
    start = time.perf_counter()
    found = []
    for i, (x, y) in enumerate(points):
        dist = np.sqrt((x - query_point[0])**2 + (y - query_point[1])**2)
        if dist <= radius:
            found.append(i)
    linear_time = time.perf_counter() - start
    
    print(f"\nLinear search: Found {len(found)} points")
    print(f"  Query time: {linear_time*1e6:.2f} microseconds")
    
    print(f"\nFor just 10 points, KDTree is {linear_time/kdtree_time:.1f}x faster!")
    print("With thousands of points, the difference is MASSIVE.")


def demo_scalability():
    """Show how speedup increases with more data."""
    print("\n\n" + "=" * 70)
    print("SCALABILITY: How performance changes with data size")
    print("=" * 70)
    
    test_sizes = [100, 500, 1000, 5000]
    query_count = 1000
    radius = 150_000
    space_size = 500_000
    
    print(f"\nTest: {query_count} queries with radius {radius:,} in {space_size:,}x{space_size:,} space")
    print("\n{:<15} {:<15} {:<15} {:<10}".format("Data Points", "Linear (ms)", "KDTree (ms)", "Speedup"))
    print("-" * 70)
    
    for n_points in test_sizes:
        # Create random points
        np.random.seed(42)
        points = np.random.rand(n_points, 2) * space_size
        
        # Create random query points
        query_points = np.random.rand(query_count, 2) * space_size
        
        # Time linear search
        start = time.perf_counter()
        for qp in query_points:
            found = []
            for pt in points:
                dist_sq = (pt[0] - qp[0])**2 + (pt[1] - qp[1])**2
                if dist_sq <= radius**2:
                    found.append(pt)
        linear_time = time.perf_counter() - start
        
        # Time KDTree search
        kdtree = KDTree(points)
        start = time.perf_counter()
        for qp in query_points:
            indices = kdtree.query_ball_point(qp, radius)
        kdtree_time = time.perf_counter() - start
        
        speedup = linear_time / kdtree_time
        print(f"{n_points:<15} {linear_time*1000:<15.2f} {kdtree_time*1000:<15.2f} {speedup:<10.2f}x")
    
    print("\nConclusion: Speedup increases dramatically with more data points!")


def demo_smoother_integration():
    """Show how to integrate KDTree into the smoother."""
    print("\n\n" + "=" * 70)
    print("INTEGRATION: Using KDTree in your Smoother class")
    print("=" * 70)
    
    code = '''
# In your Smoother.__init__:
def __init__(self, point_data, grid_settings, half_distance, search_radius):
    self.point_data = point_data
    # ... other initialization ...
    
    # Build spatial index
    self.point_coords = np.array([
        loc.location_data.point for loc in point_data
    ])
    self.point_rates = np.array([
        loc.rate for loc in point_data  
    ])
    self.kdtree = KDTree(self.point_coords)


# In your Smoother.smooth() method:
def smooth(self):
    for row in range(self.grid_settings.rows):
        for col in range(self.grid_settings.cols):
            grid_center = [x_coord + cell_half, y_coord + cell_half]
            
            # OLD WAY: Check all points (O(n))
            # for data_point in self.point_data:
            #     distance = calc_distance(grid_center, data_point)
            
            # NEW WAY: Query spatial index (O(log n))
            indices = self.kdtree.query_ball_point(grid_center, self.search_radius)
            
            # Get nearby points (vectorized!)
            nearby_coords = self.point_coords[indices]
            nearby_rates = self.point_rates[indices]
            
            # Calculate distances (vectorized!)
            dx = nearby_coords[:, 0] - grid_center[0]
            dy = nearby_coords[:, 1] - grid_center[1]
            distances_squared = dx*dx + dy*dy
            
            # Calculate weights (vectorized!)
            weights = 1.0 / (1.0 + distances_squared / half_distance_squared)
            
            # Final result
            smoothed_rate = np.sum(nearby_rates * weights) / np.sum(weights)
'''
    
    print("\nKey changes:")
    print("1. Build KDTree once in __init__ (one-time cost)")
    print("2. Use query_ball_point() instead of checking all points")
    print("3. Use numpy vectorization for distance calculations")
    print("4. Avoid creating intermediate objects (LocationFound, etc.)")
    
    print("\nCode example:")
    print(code)
    
    print("\nBenefits:")
    print("  ✓ Much faster for large grids")
    print("  ✓ Vectorized numpy operations")
    print("  ✓ Less memory allocation")
    print("  ✓ Scales better with data size")


if __name__ == "__main__":
    demo_basic_usage()
    demo_scalability()
    demo_smoother_integration()
    
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Spatial indexing with KDTree gives you:

1. FASTER QUERIES: O(log n) instead of O(n) per grid cell
2. BETTER SCALING: Performance advantage grows with data size
3. VECTORIZATION: Use numpy arrays for even more speed
4. LESS MEMORY: Avoid creating temporary objects

For your 1000x1000 grid with 100 points:
- Without index: ~100 million operations
- With KDTree: ~6-7 million operations (15-20x reduction!)

TIP: The real-world speedup depends on:
- Search radius (larger radius = more points to check)
- Point distribution (clustered vs uniform)
- Grid size vs number of points ratio
""")
    print("=" * 70)
