from enum import Enum
import time

import numpy as np
import sympy

VALID_TRAJECTORY_TYPES = {
    'jaJAj': 'PosTrapNegTrap',
    'jaJj': 'PosTrapNegTri',
    'jJAj': 'PosTriNegTrap',
    'jJj': 'PosTriNegTri',
    'JAjaJ': 'NegTrapPosTrap',
    'JAjJ': 'NegTrapPosTri',
    'JjaJ': 'NegTriPosTrap',
    'JjJ': 'NegTriPosTri',
}


def generate_motion_equations(jerk_directions):
    j_max = sympy.Symbol('j_max')
    jerk_arr = []
    for dir in jerk_directions:
        jerk_arr.append({1: j_max, 0: 0.0, -1: -j_max}[dir])
    delta_symbols = [sympy.Symbol('delta_{}'.format(i)) for i in range(len(jerk_arr))]
    position = sympy.Symbol('p_start')
    velocity = sympy.Symbol('v_start')
    acceleration = sympy.Symbol('a_start')
    for i, jerk in enumerate(jerk_arr):
        delta = delta_symbols[i]
        position += velocity * delta + sympy.Rational(1, 2) * acceleration * delta ** 2 + \
                    sympy.Rational(1, 6) * jerk * delta ** 3
        velocity += acceleration * delta + sympy.Rational(1, 2) * jerk * delta ** 2
        acceleration += jerk * delta

    motion_equations = sympy.Matrix([
        sympy.Eq(sympy.Symbol('p_end'), position),
        sympy.Eq(sympy.Symbol('v_end'), velocity),
        sympy.Eq(sympy.Symbol('a_end'), acceleration)])
    return motion_equations


def solve_seven_segment(
        segment_types, v_max, a_max, j_max, p_start, v_start, a_start, p_end, v_end, a_end):
    deltas = np.tile(np.nan, len(segment_types))
    intermediate_states = np.tile(np.nan, (len(segment_types) + 1, 3))
    intermediate_states[0, :] = (p_start, v_start, a_start)
    intermediate_states[-1, :] = (p_end, v_end, a_end)

    # Compute first/last segment durations for a_max and -a_max cases.
    if segment_types[1] == 'a':
        deltas[0] = (a_max - intermediate_states[0, 2]) / j_max
    elif segment_types[1] == 'A':
        deltas[0] = (-a_max - intermediate_states[0, 2]) / -j_max
    if segment_types[-2] == 'a':
        deltas[-1] = (intermediate_states[-1, 2] - a_max) / j_max
    elif segment_types[-2] == 'A':
        deltas[0] = (intermediate_states[-1, 2] + a_max) / -j_max

    # If there is a max velocity segment, solve for the segments next to it

# At import tarrayime, we compute the basic motion equations.
p_end_eq, v_end_eq, a_end_eq = generate_motion_equations([1, -1, 1])
delta_symbols = sympy.symbols('delta_0, delta_1, delta_2')
param_symbols = sympy.symbols('j_max, p_start, v_start, a_start, p_end, v_end, a_end')
p_end_lambda = sympy.lambdify(delta_symbols + param_symbols, p_end_eq.rhs)
v_end_lambda = sympy.lambdify(delta_symbols + param_symbols, v_end_eq.rhs)
a_end_lambda = sympy.lambdify(delta_symbols + param_symbols, a_end_eq.rhs)
# Solve acceleration equation for delta_0. The acceleration equation is linear,
# so this only has one solution.
delta_0_solution = sympy.solve(a_end_eq, delta_symbols[0])[0]
delta_0_lambda = sympy.utilities.lambdify(
    (delta_symbols[1], delta_symbols[2]) + param_symbols, delta_0_solution)
# Plug in solution for delta_0 into velocity equation, then solve for delta_1. The velocity
# equations is quadratic, so there are at most two solutions.
#
# Note: is one of the solutions always negative? Could we remove that one here to simplify
# things later?
delta_1_solutions = sympy.solve(
    v_end_eq.subs(delta_symbols[0], delta_0_solution), delta_symbols[1])
delta_1_lambdas = [sympy.utilities.lambdify((delta_symbols[2],) + param_symbols, solution)
                   for solution in delta_1_solutions]


def compute_delta_0_and_1(delta_2, j_max, p_start, v_start, a_start, p_end, v_end, a_end):
    for delta_1_lambda in delta_1_lambdas:
        delta_1 = delta_1_lambda(delta_2, j_max, p_start, v_start, a_start, p_end, v_end, a_end)
        delta_0 = delta_0_lambda(delta_1, delta_2, j_max, p_start, v_start,
                                 a_start, p_end, v_end, a_end)
        if delta_0 >= 0 and delta_1 >= 0:
            return delta_0, delta_1
    raise RuntimeError(f'No valid delta_0 and delta_1 values for delta_2={delta_2}')


def solve(v_max, a_max, j_max, p_start, v_start, a_start, p_end, v_end, a_end,
          tolerance=1e-5, max_iterations=100):
    optimization_start_time = time.time()
    iterations = []
    params = (j_max, p_start, v_start, a_start, p_end, v_end, a_end)
    delta_2_min = 0.0
    delta_2_max = 2.0 * a_max / j_max
    for iteration_i in range(max_iterations):
        delta_2 = (delta_2_min + delta_2_max) / 2.0
        delta_0, delta_1 = compute_delta_0_and_1(delta_2, *params)
        p_end_calculated = p_end_lambda(delta_0, delta_1, delta_2, *params)
        iterations.append((delta_2, p_end_calculated))
        position_error = np.abs(p_end_calculated - p_end)
        if position_error <= tolerance:
            optimization_end_time = time.time()
            optimization_info = {
                'duration': optimization_end_time - optimization_start_time,
                'iterations': iterations,
                'position_error': position_error
            }
            return (delta_0, delta_1, delta_2), optimization_info

        if p_end_calculated < p_end:
            delta_2_min = delta_2
        else:
            delta_2_max = delta_2

    raise RuntimeError(f'Error of {position_error} too high after {max_iterations} iterations')
