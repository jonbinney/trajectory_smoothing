import numpy as np
import scipy.optimize
import sympy


def generate_trajectory_equations(jerk_directions):
    j_max = sympy.Symbol('j_max')
    jerk_arr = []
    for dir in jerk_directions:
        jerk_arr.append({1: j_max, 0: 0.0, -1: -j_max}[dir])
    delta_variables = [sympy.Symbol('delta_{}'.format(i)) for i in range(len(jerk_arr))]
    position = sympy.Symbol('p_start')
    velocity = sympy.Symbol('v_start')
    acceleration = sympy.Symbol('a_start')
    for i, jerk in enumerate(jerk_arr):
        delta = delta_variables[i]
        position += velocity * delta + sympy.Rational(1, 2) * acceleration * delta ** 2 + \
                    sympy.Rational(1, 6) * jerk * delta ** 3
        velocity += acceleration * delta + sympy.Rational(1, 2) * jerk * delta ** 2
        acceleration += jerk * delta

    state_eq = sympy.Matrix([
        sympy.Eq(sympy.Symbol('p_end'), position),
        sympy.Eq(sympy.Symbol('v_end'), velocity),
        sympy.Eq(sympy.Symbol('a_end'), acceleration)])
    return delta_variables, state_eq


class SegmentedTrajectory:
    def __init__(self,
            v_max, a_max, j_max,
            jerk_directions,
            p_start, v_start, a_start,
            p_end, v_end, a_end):
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.jerk_directions = jerk_directions
        self.p_start = p_start
        self.v_start = v_start
        self.a_start = a_start
        self.p_end = p_end
        self.v_end = v_end
        self.a_end = a_end
        self.delta_values = None  # Unknown until self.optimize() is called.
        self.delta_variables, self.state_eq = generate_trajectory_equations(self.jerk_directions)
        self.subs = {
            # We don't need v_max and a_max here because they are unnecessary once we know
            # the basic shape of the acceleration, which is implied by the jerk directions.
            sympy.Symbol('j_max'): self.j_max,
            sympy.Symbol('p_start'): self.p_start,
            sympy.Symbol('v_start'): self.v_start,
            sympy.Symbol('a_start'): self.a_start,
            sympy.Symbol('p_end'): self.p_end,
            sympy.Symbol('v_end'): self.v_end,
            sympy.Symbol('a_end'): self.a_end,
        }
        self.state_lambda = sympy.lambdify(self.delta_variables, self.state_eq.subs(self.subs))
        self.state_jacobian_eq = self.state_eq.subs(self.subs).jacobian(self.delta_variables)
        self.state_jacobian_lambda = sympy.lambdify(self.delta_variables, self.state_jacobian_eq)

    def compute_end_state(self, delta_values):
        return self.state_lambda(*delta_values).flatten()

    def error_jacobian(self, x):
        delta_values = np.asarray(x) ** 2
        return self.state_jacobian_lambda(*delta_values)

    def error_function(self, x):
        x = np.asarray(x)

        # Durations must be positive.
        delta_values = x ** 2

        p_end_est, v_end_est, a_end_est = self.compute_end_state(delta_values)

        errors = np.array((p_end_est - self.p_end, v_end_est - self.v_end, a_end_est - self.a_end))
        if not np.isfinite(errors).all():
            raise RuntimeError('Non-finite values in trajectory error')
        return errors

    def optimize(self, tolerance=1e-6):
        delta_values_guess = np.random.random(len(self.delta_variables))
        solution = scipy.optimize.root(
            self.error_function, delta_values_guess, jac=self.error_jacobian, method='hybr',
            tol=tolerance)
        if not solution.success:
            raise RuntimeError('Optimization failed')
        self.delta_values = solution.x ** 2

    def optimize2(self):
        subs = {
            sympy.Symbol('j_max'): self.j_max,
            sympy.Symbol('p_start'): self.p_start,
            sympy.Symbol('v_start'): self.v_start,
            sympy.Symbol('a_start'): self.a_start,
            sympy.Symbol('delta_2'): 1.0,
        }
        concrete_state_eq = self.state_eq.subs(subs)
        print(sympy.pretty(self.state_eq[0], use_unicode=True))
        print(sympy.pretty(self.state_eq[1], use_unicode=True))
        print(sympy.pretty(self.state_eq[2], use_unicode=True))
        print('sdlkfj'*30)
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        print(columns)

        # Solve acceleration equation for delta_1