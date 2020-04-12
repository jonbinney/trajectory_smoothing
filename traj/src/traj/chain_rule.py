import sympy

s = sympy.Symbol('s')


def chain_rule(path_functions, state):
    """

    Args:
        path_functions:
        state:

    Returns:

    """
    return [
        path_functions[0],
        path_functions[1] * state[1],
        path_functions[2] * state[1] ** 2 + path_functions[1] * state[2],
        path_functions[3] * state[1] ** 3 + 3 * path_functions[2] * state[2] * state[1] + path_functions[1] * state[3]
    ]


def evaluate(trajectory_functions, s):
    return [f.subs({s: s})]


def is_state_valid(path_functions, state, min_values, max_values):
    """
    Check whether a position, velocity, acceleration, and jerk of s falls within joint limits for the
    individual joints. This depends on the shape of the path through joint space.

    For an explanation of how the chain rule is used here to combine the equation of the path with the derivatives
    of s(t), see equations (3) and (10) in the following paper.

        Debrouwere, Frederik & Van Loock, Wannes & Pipeleers, Goele & Tran-Dinh, Quoc & Diehl, Moritz & Schutter,
        Joris & Swevers, Jan. (2013). "Time-Optimal Path Following for Robots with Trajectory Jerk Constraints using
        Sequential Convex Programming"

    Args:
        path_functions: sequence of length 4 where path_functions[i] is the i'th derivative of joint
          positions with respect to s: [q(s), dq / ds, d^2q / ds^2, d^3q / ds^2]
        state: sequence of length 4 where state[i] is the i'th derivative of s with respect to t:
            [s(t), ds / dt, d^2s / dt^2, d^3s / dt^2]
        min_values: sequence of length 4 where min_values[i] is a vector of minimum values for the i'th derivative
            of q(s) with respect to t.
        max_values: sequence of length 4 where max_values[i] is a vector of maximum values for the i'th derivative
            of q(s) with respect to t.

    Returns: True if the state is within limits, false otherwise.
    """
    pass
