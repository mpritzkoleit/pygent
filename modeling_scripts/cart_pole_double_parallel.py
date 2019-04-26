import os
import sympy as sp
import numpy as np
from sympy import sin, cos, Function
try:
    from sympy_to_c import sympy_to_c as sp2c
except ImportError:
    print('sympy-to-c could not be imported!')
import pickle

def modeling():
    t = sp.Symbol('t') # time
    params = sp.symbols('m0, m1, m2, J1, J2, l1, l2, g, d0, d1, d2') # system parameters
    m0, m1, m2, J1, J2, l1, l2, g, d0, d1, d2 = params
    params_values = [(m0, 3.34), (m1, 0.3583), (m2, 0.5383), (J1, 0.0379999), (J2, 0.12596),
                     (l1, 0.5), (l2, 0.75), (g, 9.81), (d0, 0.1), (d1, 0.006588), (d2, 0.006588)]
    # force
    F = sp.Symbol('F')

    # generalized coordinates
    q0_t = Function('q0')(t)
    dq0_t = q0_t.diff(t)
    ddq0_t = q0_t.diff(t, 2)
    q1_t = Function('q1')(t)
    dq1_t = q1_t.diff(t)
    ddq1_t = q1_t.diff(t, 2)
    q2_t = Function('q2')(t)
    dq2_t = q2_t.diff(t)
    ddq2_t = q2_t.diff(t, 2)

    # position vectors
    p0 = sp.Matrix([q0_t, 0])
    p1 = sp.Matrix([q0_t - 0.5*l1*sin(q1_t),  0.5*l1*cos(q1_t)])
    p2 = sp.Matrix([q0_t -  0.5*l2*sin(q2_t),  0.5*l2*cos(q2_t)])

    # velocity vectors
    dp0 = p0.diff(t)
    dp1 = p1.diff(t)
    dp2 = p2.diff(t)

    # kinetic energy T
    T0 = m0/2*(dp0.T*dp0)[0]
    T1 = (m1*(dp1.T*dp1)[0] + J1*dq1_t**2)/2
    T2 = (m2*(dp2.T*dp2)[0] + J2*dq2_t**2)/2
    T = T0 + T1 + T2

    # potential energy V
    V = m1*g*p1[1] + m2*g*p2[1]

    # lagrangian L
    L = T - V
    L = L.expand()
    L = sp.trigsimp(L)

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i = Q_i

    Q0 = F - d0*dq0_t
    Q1 =   - d1*dq1_t
    Q2 =   - d2*dq2_t

    Eq0 = L.diff(dq0_t, t) - L.diff(q0_t) - Q0 # = 0
    Eq1 = L.diff(dq1_t, t) - L.diff(q1_t) - Q1 # = 0
    Eq2 = L.diff(dq2_t, t) - L.diff(q2_t) - Q2 # = 0

    # equations of motion
    Eq = sp.Matrix([Eq0, Eq1, Eq2])

    ddq_t = sp.Matrix([ddq0_t, ddq1_t, ddq2_t])
    M = Eq.jacobian(ddq_t)

    q_zeros = [(ddq0_t, 0), (ddq1_t, 0), (ddq2_t, 0)]
    ddq = M.inv()* -Eq.subs(q_zeros)

    # state space model

    # functions of x, u
    x1_t = sp.Function('x1')(t)
    x2_t = sp.Function('x2')(t)
    x3_t = sp.Function('x3')(t)
    x4_t = sp.Function('x4')(t)
    x5_t = sp.Function('x5')(t)
    x6_t = sp.Function('x6')(t)
    x_t = sp.Matrix([x1_t, x2_t, x3_t, x4_t, x5_t, x6_t])

    u_t = sp.Function('u')(t)

    # symbols of x, u
    x1, x2, x3, x4, x5, x6, u = sp.symbols("x1, x2, x3, x4, x5, x6, u")
    xx = [x1, x2, x3, x4, x5, x6]

    # replace generalized coordinates with states
    xu_subs = [(dq0_t, x4_t), (dq1_t, x5_t), (dq2_t, x6_t), (q0_t, x1_t), (q1_t, x2_t), (q2_t, x3_t), (F, u_t)]
    ddq = ddq.subs(xu_subs)

    # first order ODE (right hand side)
    dx_t = sp.Matrix([x4_t, x5_t, x6_t, ddq[0], ddq[1], ddq[2]])

    # linearized dynamics
    A = dx_t.jacobian(x_t)
    B = dx_t.diff(u_t)

    # symbolic expressions of A and B with parameter values
    Asym = A.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values)
    Bsym = B.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values)

    # callable functions
    A_func = sp.lambdify((x2, x3, x4, x5, x6, u), Asym, modules="numpy")
    B_func = sp.lambdify((x2, x3), Bsym, modules="numpy")

    dx_t_sym = dx_t.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values) # replacing all symbolic functions with symbols

    # RHS as callable function
    try: # use c-code
        dx_c_func = sp2c.convert_to_c((x1, x2, x3, x4, x5, x6, u), dx_t_sym, cfilepath="c_files/cart_pole_double_parallel.c",
                                      use_exisiting_so=False)
        dxdt = lambda t, x, u: dx_c_func(*x, *u).T[0]

    except:
        print('C-function of systems ODE could not be created')
        dx_func = sp.lambdify((x1, x2, x3, x4, x5, x6, u), dx_t_sym[:], modules="numpy")  # creating a callable python function
        dxdt = lambda t, x, u: np.array(dx_func(*x, *u))
    
    return dxdt

def load_existing():
    try:
        x1, x2, x3, x4, x5, x6, u = sp.symbols("x1, x2, x3, x4, x5, x6, u")
        dx_t_sym = sp.Matrix([[0], [0], [0], [0], [0], [0]])
        dx_c_func = sp2c.convert_to_c((x1, x2, x3, x4, x5, x6, u), dx_t_sym, cfilepath="c_files/cart_pole_double_parallel.c",
                                      use_exisiting_so=True)
        dxdt = lambda t, x, u: dx_c_func(*x, *u).T[0]
        assert (any(dxdt(0, [0, 0, 0, 1., 1., 1.], [0]) != [0., 0., 0., 0., 0., 0.]))
        print('Model loaded')
    except:
        print('Model could not be loaded! Rerunning model creation!')
        dxdt = modeling()
    return dxdt

if __name__ == "__main__":
    # execute only if run as a script
    modeling()