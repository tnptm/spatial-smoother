from smoother import runner as smoothie_runner 
#from distance import run_distance


def main():
    print("This runs simple speed comparison for distance and distance-population weighting!")
    #distance.run_distance(plot=False)
    #distance_population.run_dist_pop_weight(plot=False)

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
    #smoothie_runner.run(distance_interp_settings)

    # print numpy version
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
    #smoothie_runner.run(distance_interp_settings_numpy)

    # print numba njit version
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
    #smoothie_runner.run(dist_pop_interp_settings_wpop)

if __name__ == "__main__":
    main()
