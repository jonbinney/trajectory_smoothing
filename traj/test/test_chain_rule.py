import sympy

from traj import chain_rule

s = sympy.Symbol('s')
t = sympy.Sumbol('t')



def test_linear_function_chain_rule():
    f_of_x = 2.0 * s
    g_of_x = sympy.Float(t)
    f_of_g_of_x = f_of_x.subs({s: g_of_x})
    path_functions = [f_of_x]
    for derivative_i in range(3):
        path_functions.append([path_functions[-1].diff(s)])
    s_test = 1.0
    state = [1.0, 1.0, 1.0, 1.0]
    print(chain_rule.chain_rule(path_functions, state))

