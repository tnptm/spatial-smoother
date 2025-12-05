from smoother import runner as smoothie_runner 
#from distance import run_distance


def main():
    """Main function to run distance and distance-population weighting comparisons."""

    print("This runs simple speed comparison for distance and distance-population weighting!")
    
    # print pure python version (7.5s)
    distance_interp_settings = smoothie_runner.RunnerSettings(
        smooth_func='distance',
        plot=True,
        half_distance=10_000,
        search_radius=150_000,
        nrows=1000,
        ncols=1000,
        grid_size=500,
        num_points=100,
        max_population=200_000,
        save_smoothed=True,
        print_all=False,
        looper_func='pure-python-looper'
    )
    smoothie_runner.run(distance_interp_settings)

    # print numpy version (>2x slower than pure python looper for this case)
    distance_interp_settings_numpy = smoothie_runner.RunnerSettings(
        smooth_func='distance',
        plot=True,
        half_distance=10_000,
        search_radius=150_000,
        nrows=1000,
        ncols=1000,
        grid_size=500,
        num_points=100,
        max_population=200_000,
        save_smoothed=True,
        print_all=False,
        looper_func='numpy-vectorized'
    )
    smoothie_runner.run(distance_interp_settings_numpy)

    # print numba njit version (0.4x speedup over pure python looper 7.5s -->3.5s)
    distance_interp_settings_njit = smoothie_runner.RunnerSettings(
        smooth_func='distance',
        plot=True,
        half_distance=10_000,
        search_radius=150_000,
        nrows=1000,
        ncols=1000,
        grid_size=500,
        num_points=100,
        max_population=200_000,
        save_smoothed=True,
        print_all=False,
        looper_func='numba-jit'
    )
    smoothie_runner.run(distance_interp_settings_njit)

    dist_pop_interp_settings_wpop = smoothie_runner.RunnerSettings(
        smooth_func='distance-population',
        #plot=True,
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
    smoothie_runner.run(dist_pop_interp_settings_wpop)

if __name__ == "__main__":
    main()
