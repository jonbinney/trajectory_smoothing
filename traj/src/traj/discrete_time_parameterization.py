import numpy as np

MAX_TIME_STEPS = 10000

# How close we have to be to a given position/velocity/acceleration to consider it "reached".
# These are needed because we are using a discrete approximation of the trajectory. These
# should all be small enough that if you commanded zero velocity starting now, the resulting
# deceleration would be less than the robot's limits.
POSITION_THRESHOLD = 0.01
VELOCITY_THRESHOLD = 0.01
ACCELERATION_THRESHOLD = 0.01
JERK_THRESHOLD = 0.01


def compute_stopping_trajectory(p_start, v_start, a_start, is_valid, j_max, delta_t):
    positions = np.zeros(MAX_TIME_STEPS)
    velocities = np.zeros(MAX_TIME_STEPS)
    accelerations = np.zeros(MAX_TIME_STEPS)
    jerks = np.zeros(MAX_TIME_STEPS)
    positions[0] = p_start
    velocities[0] = v_start
    accelerations[0] = a_start

    # Decelerate until our velocity drops to zero.
    for time_i in range(1, MAX_TIME_STEPS):
        if velocities[time_i - 1] < VELOCITY_THRESHOLD:
            break
        elif time_i == MAX_TIME_STEPS - 1:
            raise RuntimeError('Failed to find a solution after {} trajectory points'.format(
                MAX_TIME_STEPS))

        best_next_jerk = None
        for next_jerk in (0.0, -j_max):
            next_acceleration = accelerations[time_i - 1] + next_jerk * delta_t
            next_velocity = velocities[time_i - 1] + next_acceleration * delta_t
            next_position = positions[time_i - 1] + next_velocity * delta_t

            # Check instantaneous limits.
            if not is_valid(next_position, next_velocity, next_acceleration, next_jerk):
                continue

            best_next_jerk = next_jerk

        if best_next_jerk is None:
            return None

        jerks[time_i] = best_next_jerk
        accelerations[time_i] = accelerations[time_i - 1] + jerks[time_i] * delta_t
        velocities[time_i] = velocities[time_i - 1] + accelerations[time_i] * delta_t
        positions[time_i] = positions[time_i - 1] + velocities[time_i] * delta_t

    # This is the timestep at which we reach zero velocity if use the most negative valid jerk
    # at every timestep. Unfortunately we probably reach zero velocity with a large negative
    # acceleration. We need to reach zero velocity and zero acceleration at the same time,
    # and so need to switch from max negative jerk to max positive jerk at some timestep.
    # the next loop searches for that timestep.
    soonest_velocity_zero_time = time_i - 1

    if accelerations[soonest_velocity_zero_time] > -ACCELERATION_THRESHOLD:
        return (positions[1:time_i],
                velocities[1:time_i],
                accelerations[1:time_i],
                jerks[1:time_i])

    # We need to add a positive jerk section.
    for positive_jerk_start_time_i in range(soonest_velocity_zero_time, 0, -1):
        for time_i in range(positive_jerk_start_time_i, MAX_TIME_STEPS):
            if accelerations[time_i - 1] > -ACCELERATION_THRESHOLD:
                return (positions[1:time_i], velocities[1:time_i], accelerations[1:time_i],
                        jerks[1:time_i])
            elif velocities[time_i - 1] < -VELOCITY_THRESHOLD:
                # We weren't reduce acceleration magnitude to zero before velocity hit zero.
                break

            jerks[time_i] = j_max
            accelerations[time_i] = accelerations[time_i - 1] + jerks[time_i] * delta_t
            velocities[time_i] = velocities[time_i - 1] + accelerations[time_i] * delta_t
            positions[time_i] = positions[time_i - 1] + velocities[time_i] * delta_t

            if not is_valid(positions[time_i], velocities[time_i],
                            accelerations[time_i], jerks[time_i]):
                break

    # We were unable to decelerate.
    return None


def parameterize_path_discrete(p_start, p_end, is_valid, j_max, delta_t):
    positions = np.zeros(MAX_TIME_STEPS)
    velocities = np.zeros(MAX_TIME_STEPS)
    accelerations = np.zeros(MAX_TIME_STEPS)
    jerks = np.zeros(MAX_TIME_STEPS)
    positions[0] = p_start
    velocities[0] = 0.0
    accelerations[0] = 0.0
    jerks[0] = 0.0

    # Stopping trajectory for each timestep. We only save this for debugging, so that we can see
    # why the algorithm made the choice it did at each timestep.
    stopping_trajectories = {}

    for time_i in range(1, MAX_TIME_STEPS):
        if positions[time_i - 1] >= p_end - POSITION_THRESHOLD:
            # Reached our goal.
            stopping_positions, stopping_velocities, stopping_accelerations, stopping_jerks = \
                stopping_trajectories[time_i - 1]
            return (
                np.hstack((positions[:time_i], stopping_positions)),
                np.hstack((velocities[:time_i], stopping_velocities)),
                np.hstack((accelerations[:time_i], stopping_accelerations)),
                np.hstack((jerks[:time_i], stopping_jerks)))

        best_next_jerk = None

        for next_jerk in [-j_max, 0.0, j_max]:
            next_acceleration = accelerations[time_i - 1] + next_jerk * delta_t
            next_velocity = velocities[time_i - 1] + next_acceleration * delta_t
            next_position = positions[time_i - 1] + next_velocity * delta_t

            if not is_valid(next_position, next_velocity, next_acceleration, next_jerk):
                continue

            stopping_trajectory = compute_stopping_trajectory(
                next_position, next_velocity, next_acceleration,
                is_valid, j_max, delta_t)
            if stopping_trajectory is None:
                # There will be no valid way to stop if we apply this jerk.
                continue

            stopping_trajectories[time_i] = stopping_trajectory

            best_next_jerk = next_jerk

        if best_next_jerk is None:
            raise RuntimeError('No valid jerk found for timestep {}'.format(time_i))

        jerks[time_i] = best_next_jerk
        accelerations[time_i] = accelerations[time_i - 1] + best_next_jerk * delta_t
        velocities[time_i] = velocities[time_i - 1] + accelerations[time_i] * delta_t
        positions[time_i] = positions[time_i - 1] + velocities[time_i] * delta_t

    raise RuntimeError('Failed to find a solution after {} trajectory points'.format(MAX_TIME_STEPS))
