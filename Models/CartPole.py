import os
import sympy as sp
import numpy as np
from sympy import sin, cos, Function
try:
    from sympy_to_c import sympy_to_c as sp2c
except ImportError:
    print('sympy-to-c could not be imported!')
import pickle
#import dill
def modeling():
    t = sp.Symbol('t') # time
    params = sp.symbols('m0, m1, J1, l1, g, d0, d1') # system parameters
    m0, m1, J1, l1, g, d0, d1 = params
    params_values = [(m0, 3.34), (m1, 0.3583), (J1, 0.0379999),
                     (l1, 0.5), (g, 9.81), (d0, 0.1), (d1, 0.006588)]
    # force
    F = sp.Symbol('F')

    # generalized coordinates
    q0_t = Function('q0')(t)
    dq0_t = q0_t.diff(t)
    ddq0_t = q0_t.diff(t, 2)
    q1_t = Function('q1')(t)
    dq1_t = q1_t.diff(t)
    ddq1_t = q1_t.diff(t, 2)

    # position vectors
    p0 = sp.Matrix([q0_t, 0])
    p1 = sp.Matrix([q0_t - l1*sin(q1_t), l1*cos(q1_t)])

    # velocity vectors
    dp0 = p0.diff(t)
    dp1 = p1.diff(t)

    # kinetic energy T
    T0 = m0/2*(dp0.T*dp0)[0]
    T1 = (m1*(dp1.T*dp1)[0] + J1*dq1_t**2)/2
    T = T0 + T1

    # potential energy V
    V = m1*g*p1[1]

    # lagrangian L
    L = T - V
    L = L.expand()
    L = sp.trigsimp(L)

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i = Q_i

    Q0 = F - d0/2*dq0_t**2
    Q1 =   - d1/2*dq1_t**2

    Eq0 = L.diff(dq0_t, t) - L.diff(q0_t) - Q0 # = 0
    Eq1 = L.diff(dq1_t, t) - L.diff(q1_t) - Q1 # = 0
    # equations of motion
    Eq = sp.Matrix([Eq0, Eq1])

    ddq_t = sp.Matrix([ddq0_t, ddq1_t])
    M = Eq.jacobian(ddq_t)

    q_zeros = [(ddq0_t, 0), (ddq1_t, 0)]
    ddq = M.inv()* -Eq.subs(q_zeros)

    # state space model

    # functions of x, u
    x1_t = sp.Function('x1')(t)
    x2_t = sp.Function('x2')(t)
    x3_t = sp.Function('x3')(t)
    x4_t = sp.Function('x4')(t)
    x_t = sp.Matrix([x1_t, x2_t, x3_t, x4_t])

    u_t = sp.Function('u')(t)

    # symbols of x, u
    x1, x2, x3, x4, u = sp.symbols("x1, x2, x3, x4, u")
    xx = [x1, x2, x3, x4]

    # replace generalized coordinates with states
    xu_subs = [(dq0_t, x3_t), (dq1_t, x4_t), (q0_t, x1_t), (q1_t, x2_t), (F, u_t)]
    ddq = ddq.subs(xu_subs)

    # first order ODE (right hand side)
    dx_t = sp.Matrix([x3_t, x4_t, ddq[0], ddq[1]])

    # linearized dynamics
    A = dx_t.jacobian(x_t)
    B = dx_t.diff(u_t)

    # symbolic expressions of A and B with parameter values
    Asym = A.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values)
    Bsym = B.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values)

    # callable functions
    A_func = sp.lambdify((x1, x2, x3, x4, u), Asym, modules="numpy")
    B_func = sp.lambdify((x1, x2, x3, x4, u), Bsym, modules="numpy")

    dx_t_sym = dx_t.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values) # replacing all symbolic functions with symbols

    # RHS as callable function
    try: # use c-code
        dx_c_func = sp2c.convert_to_c((x1, x2, x3, x4, u), dx_t_sym, cfilepath="cartPole.c",
                                  use_exisiting_so=False)

        dxdt = lambda t, x, u: dx_c_func(*x, *u).T[0]

    except:
        print('C-function of systems ODE could not be created')
        dx_func = sp.lambdify((x1, x2, x3, x4, u), dx_t_sym[:], modules="numpy")  # creating a callable python function
        dxdt = lambda t, x, u: np.array(dx_func(*x, *u))


    return dxdt

if __name__ == "__main__":
    # execute only if run as a script
    modeling()