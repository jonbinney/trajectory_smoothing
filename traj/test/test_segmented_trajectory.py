import time
import numpy as np

from traj import segmented_trajectory

def compare_solution_to_prediction(
        jerk_directions, v_max, a_max, j_max, p_start, v_start, a_start,
        p_end, v_end, a_end, actual_deltas):
    calculated_deltas, optimization_info = segmented_trajectory.solve(
        v_max, a_max, j_max, p_start, v_start, a_start, p_end, v_end, a_end)
    print(f'Solved for trajectory in {int(round(optimization_info["duration"]*1e6))}us'
          + f'({len(optimization_info["iterations"])} iterations)')
    params = (j_max, p_start, v_start, a_start, p_end, v_end, a_end)
    p_end_calculated = segmented_trajectory.p_end_lambda(*calculated_deltas, *params)
    v_end_calculated = segmented_trajectory.v_end_lambda(*calculated_deltas, *params)
    a_end_calculated = segmented_trajectory.a_end_lambda(*calculated_deltas, *params)

    assert np.isclose(p_end_calculated, p_end, atol=1e5)
    assert np.isclose(v_end_calculated, v_end, atol=1e5)
    assert np.isclose(a_end_calculated, a_end, atol=1e5)

def test_nonzero_ending_velocity_and_acceleration():
    compare_solution_to_prediction(
        (1, -1, 1), 2.0, 3.0, 1.0,
        0.0, 0.0, 0.0,
        2.166666666666667, 1.5, 1.0,
        actual_deltas=(1., 1., 1.))

def test_zero_trajectory():
    compare_solution_to_prediction(
        (1, -1, 1), 2.0, 3.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, actual_deltas=(0.0, 0.0, 0.0))

def test_simple_trajectory():
    compare_solution_to_prediction(
        (1, -1, 1), 2.0, 3.0, 1.0,
        0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, actual_deltas=(1.0, 2.0, 1.0))
