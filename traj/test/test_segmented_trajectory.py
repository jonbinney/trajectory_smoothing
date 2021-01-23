import time
import numpy as np

import traj.segmented_trajectory


def test_zero_length_trajectory():
    trajectory = traj.segmented_trajectory.SegmentedTrajectory(
        1.0, [1, -1, 1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    error = trajectory.error_function([0.0, 0.0, 0.0])
    assert np.isclose(error, (0.0,)).all()

    trajectory.optimize()
    error = trajectory.error_function(trajectory.delta_values)
    assert np.isclose(error, (0.0,)).all()

    print(trajectory.delta_values)
    assert np.isclose(trajectory.delta_values, (0.0, 0.0, 0.0), atol=1e-3).all()

def test_single_segment_trajectory():
   trajectory = traj.segmented_trajectory.SegmentedTrajectory(
       1.0, [1], 0.0, 0.0, 0.0, 1./6., 0.5, 1.0)

   # With the correct duration, there should be no error.
   error = trajectory.error_function([1.0])
   assert np.allclose(error, (0.0, 0.0, 0.0))

   assert np.allclose(trajectory.compute_end_state([0.0]), (0.0, 0.0, 0.0))
   assert np.allclose(trajectory.compute_end_state([2.0]), (4./3., 2.0, 2.0))

def test_two_segment_trajectory():
   trajectory = traj.segmented_trajectory.SegmentedTrajectory(
      1.0, [1, -1], 0.0, 0.0, 0.0, 0.0, 0.5, 1.0)

   assert np.allclose(trajectory.compute_end_state([0.0, 0.0]), (0.0,  0.0, 0.0))
   assert np.allclose(trajectory.compute_end_state([1.0, 2.0]), (11./6.,  0.5, -1.0))

