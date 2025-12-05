import math
import time

import numpy as np
from numba import njit

def create_sample_points(num_points):
    return np.random.randn(num_points, 2)  # N rows, each [x, y]

def slow_python(points: np.ndarray):
    """Example of slow distance calculation in pure Python."""
    #import math
    #import time

    def distances_python(points: np.ndarray):
        # reserve memory for list
        out: list = [None] * len(points)
        for i, (x, y) in enumerate(points):
            out[i] = math.sqrt(x*x + y*y)
        return out

    t0 = time.time()
    distances_python(points)
    print("Python loop:", time.time() - t0, "s")

def fast_numpy(points):
    """Example of fast distance calculation using NumPy vectorization."""
    #import numpy as np
    #import time

    def distances_numpy(points):
        return np.sqrt(np.sum(points**2, axis=1))

    t0 = time.time()
    distances_numpy(points)
    print("NumPy vectorized:", time.time() - t0, "s")


@njit
def distances_numba(points):
    out = np.empty(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        x = points[i, 0]
        y = points[i, 1]
        out[i] = (x*x + y*y)**0.5
    return out


def main():
    num_points = 1_000_000
    points = create_sample_points(num_points)

    slow_python(points)
    fast_numpy(points)

    # Numba JIT compilation
    # warm-up compile
    distances_numba(points[:10])

    t0 = time.time()
    distances_numba(points)
    print("Numba JIT:", time.time() - t0, "s")

    # results as expected in sample:
    """Python loop: 0.4689948558807373 s
NumPy vectorized: 0.01546478271484375 s
Numba JIT: 0.0027680397033691406 s"""
    

if __name__ == "__main__":
    main()