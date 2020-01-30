import os
import sympy as sp
import numpy as np
from sympy import sin, cos, Function
try:
    from sympy_to_c import sympy_to_c as sp2c
except ImportError:
    print('sympy-to-c could not be imported!')
import pickle
'''
https://www.acin.tuwien.ac.at/file/publications/cds/pre_post_print/glueck2013.pdf
'''
def modeling(linearized=True):
    t = sp.Symbol('t') # time
    params = sp.symbols('m0, m1, m2, J1, J2, a1, a2, l1, l2, g, d0, d1, d2') # system parameters
    m0, m1, m2, J1, J2, a1, a2, l1, l2, g, d0, d1, d2 = params
    params_values = [(m0, 3.34), (m1, 0.876), (m2,  0.938), (J1, 0.013), (J2, 0.024),
                     (a1, 0.215), (a2, 0.269), (l1, 0.323), (l2, 0.419), (g, 9.81),
                     (d0, 0.1), (d1, 0.215), (d2, 0.002)]
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
    p1 = sp.Matrix([q0_t - a1*sin(q1_t), a1*cos(q1_t)])
    p2 = sp.Matrix([q0_t - l1*sin(q1_t) - a2*sin(q2_t), l1*cos(q1_t) + a2*cos(q2_t)])

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

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i = Q_i

    Q0 = F - d0*dq0_t
    Q1 =   - d1*dq1_t + d2*(dq2_t - dq1_t)
    Q2 =   - d2*(dq2_t - dq1_t)

    Eq0 = L.diff(dq0_t, t) - L.diff(q0_t) - Q0 # = 0
    Eq1 = L.diff(dq1_t, t) - L.diff(q1_t) - Q1 # = 0
    Eq2 = L.diff(dq2_t, t) - L.diff(q2_t) - Q2 # = 0

    # equations of motion
    Eq = sp.Matrix([Eq0, Eq1, Eq2])

    # partial linerization / acceleration as input, not force/torque
    # new input - acceleration of the cart
    a = sp.Function('a')(t)

    # replace ddq0 with a
    Eq_a = Eq.subs(ddq0_t, a)

    # solve for F
    sol = sp.solve(Eq_a, F)
    Fa = sol[F]  # F(a)

    # partial linearization
    Eq_lin = Eq.subs(F, Fa)

    # solve for ddq/dt
    ddq_t = sp.Matrix([ddq0_t, ddq1_t, ddq2_t])
    if linearized:
        ddq = sp.solve(Eq_lin, ddq_t)
    else:
        ddq = sp.solve(Eq, ddq_t)

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
    if linearized:
        xu_subs = [(dq0_t, x4_t), (dq1_t, x5_t), (dq2_t, x6_t), (q0_t, x1_t), (q1_t, x2_t), (q2_t, x3_t), (a, u_t)]
    else:
        xu_subs = [(dq0_t, x4_t), (dq1_t, x5_t), (dq2_t, x6_t), (q0_t, x1_t), (q1_t, x2_t), (q2_t, x3_t), (F, u_t)]
    # first order ODE (right hand side)
    dx_t = sp.Matrix([x4_t, x5_t, x6_t, ddq[ddq0_t], ddq[ddq1_t], ddq[ddq2_t]])
    dx_t = dx_t.subs(xu_subs)

    # linearized dynamics
    A = dx_t.jacobian(x_t)
    B = dx_t.diff(u_t)

    # symbolic expressions of A and B with parameter values
    Asym = A.subs(list(zip(x_t, xx))).subs(u_t, u)
    Bsym = B.subs(list(zip(x_t, xx))).subs(u_t, u)

    dx_t_sym = dx_t.subs(list(zip(x_t, xx))).subs(u_t, u)# replacing all symbolic functions with symbols
    print(dx_t_sym)
    if linearized:
        lin = '_lin'
    else:
        lin = ''
    with open('c_files/cart_pole_double_serial' + lin + '_ode.p', 'wb') as opened_file:
        pickle.dump(dx_t_sym, opened_file)
    with open('c_files/cart_pole_double_serial' + lin + '_A.p', 'wb') as opened_file:
        pickle.dump(Asym, opened_file)
    with open('c_files/cart_pole_double_serial' + lin + '_B.p', 'wb') as opened_file:
        pickle.dump(Bsym, opened_file)
    # RHS as callable function
    dxdt, A, B = load_existing(linearized=linearized)

    return dxdt , A, B

def load_existing(linearized=True):
    if linearized:
        lin = '_lin'
    else:
        lin = ''
    path = os.path.dirname(os.path.abspath(__file__))
    x1, x2, x3, x4, x5, x6, u = sp.symbols("x1, x2, x3, x4, x5, x6, u")
    with open(path+'/c_files/cart_pole_double_serial' + lin + '_ode.p', 'rb') as opened_file:
        dx_t_sym = pickle.load(opened_file)
        print('Model loaded')
    with open(path+'/c_files/cart_pole_double_serial' + lin + '_A.p', 'rb') as opened_file:
        Asym = pickle.load(opened_file)
        print('A matrix loaded')
    with open(path+'/c_files/cart_pole_double_serial' + lin + '_B.p', 'rb') as opened_file:
        Bsym = pickle.load(opened_file)
        print('B matrix loaded')

    params = sp.symbols('m0, m1, m2, J1, J2, a1, a2, l1, l2, g, d0, d1, d2')  # system parameters
    m0, m1, m2, J1, J2, a1, a2, l1, l2, g, d0, d1, d2 = params
    params_values = [(m0, 3.34), (m1, 0.876), (m2, 0.938), (J1, 0.013), (J2, 0.024),
                     (a1, 0.215), (a2, 0.269), (l1, 0.323), (l2, 0.419), (g, 9.81),
                     (d0, 0.1), (d1, 0.215), (d2, 0.002)]

    dx_t_sym = dx_t_sym.subs(params_values)
    Asym = Asym.subs(params_values)
    Bsym = Bsym.subs(params_values)

    try:
        dx_c_func = sp2c.convert_to_c((x1, x2, x3, x4, x5, x6, u), dx_t_sym,
                                      cfilepath=path+'/c_files/cart_pole_double_serial' + lin + '.c',
                                      use_exisiting_so=False)
        A_c_func = sp2c.convert_to_c((x1, x2, x3, x4, x5, x6, u), Asym,
                                     cfilepath=path+'/c_files/cart_pole_double_serial' + lin + '_A.c',
                                     use_exisiting_so=False)
        B_c_func = sp2c.convert_to_c((x1, x2, x3, x4, x5, x6, u), Bsym,
                                     cfilepath=path+'/c_files/cart_pole_double_serial' + lin + '_B.c',
                                     use_exisiting_so=False)
        A = lambda x, u: A_c_func(*x, *u)
        B = lambda x, u: B_c_func(*x, *u)
        dxdt = lambda t, x, u: dx_c_func(*x, *u).T[0]
        assert(any(dxdt(0, [0, 0, 0, 1., 1., 1.], [0]) != [0., 0., 0., 0., 0., 0.]))
        print('Using C-function')
    except:
        A_func = sp.lambdify((x1, x2, x3, x4, x5, x6, u), Asym, modules="numpy")
        B_func = sp.lambdify((x1, x2, x3, x4, x5, x6, u), Bsym, modules="numpy")
        A = lambda x, u: A_func(*x, *u)
        B = lambda x, u: B_func(*x, *u)

        dx_func = sp.lambdify((x1, x2, x3, x4, x5, x6, u), dx_t_sym[:], modules="numpy")  # creating a callable python function
        dxdt = lambda t, x, u: np.array(dx_func(*x, *u))
        assert(any(dxdt(0, [0, 0, 0, 1., 1., 1.], [0]) != [0., 0., 0., 0., 0., 0.]))
        print('Using lambdify')
    return dxdt, A, B

if __name__ == "__main__":
    # execute only if run as a script
    modeling(linearized=True)

