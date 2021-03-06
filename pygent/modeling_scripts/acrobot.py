import sympy as sp
import numpy as np
from sympy import sin, cos, Function
try:
    from sympy_to_c import sympy_to_c as sp2c
except ImportError:
    print('sympy-to-c could not be imported!')
import pickle
import os

def modeling(linearized=True):
    t = sp.Symbol('t') # time
    params = sp.symbols('m0, m1, J0, J1, l0, l1, g, d0, d1') # system parameters
    m0, m1, J0, J1, l0, l1, g, d0, d1 = params
    params_values = [(m0, 0.3583), (m1, 0.3583), (J0, 0.0379999), (J1, 0.0379999),
                     (l0, 0.5), (l1, 0.5), (g, 9.81), (d0, 0.006588), (d1, 0.006588)]
    # torque
    tau = sp.Symbol('tau')

    # generalized coordinates
    q0_t = Function('q0')(t)
    dq0_t = q0_t.diff(t)
    ddq0_t = q0_t.diff(t, 2)
    q1_t = Function('q1')(t)
    dq1_t = q1_t.diff(t)
    ddq1_t = q1_t.diff(t, 2)

    # position vectors of COMs
    p0 = sp.Matrix([-0.5*l0*sin(q0_t), 0.5*l0*cos(q0_t)])
    p1 = 2*p0 + sp.Matrix([- 0.5*l1*sin(q0_t+q1_t), 0.5*l1*cos(q0_t+q1_t)])

    # velocity vectors
    dp0 = p0.diff(t)
    dp1 = p1.diff(t)

    # kinetic energy T
    T0 = (m0*(dp0.T*dp0)[0] + J0*dq0_t**2)/2
    T1 = (m1*(dp1.T*dp1)[0] + J1*(dq0_t+dq1_t)**2)/2

    T = T0 + T1

    # potential energy V
    V = m0*g*p0[1] + m1*g*p1[1]

    # lagrangian L
    L = T - V
    L = L.expand()
    L = sp.trigsimp(L)

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i = Q_i

    Q0 =      - d0*dq0_t
    Q1 =  tau - d1*dq1_t

    Eq0 = L.diff(dq0_t, t) - L.diff(q0_t) - Q0 # = 0
    Eq1 = L.diff(dq1_t, t) - L.diff(q1_t) - Q1 # = 0
    # equations of motion
    Eq = sp.Matrix([Eq0, Eq1])

    # partial linerization / acceleration as input, not force/torque
    # new input - acceleration of the cart
    a = sp.Function('a')(t)

    # replace ddq0 with a
    Eq_a = Eq.subs(ddq1_t, a)

    # solve for F
    sol = sp.solve(Eq_a, tau)
    tau_a = sol[tau]  # F(a)

    # partial linearization
    Eq_lin = Eq.subs(tau, tau_a)

    # solve for ddq/dt
    ddq_t = sp.Matrix([ddq0_t, ddq1_t])
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
    x_t = sp.Matrix([x1_t, x2_t, x3_t, x4_t])

    u_t = sp.Function('u')(t)

    # symbols of x, u
    x1, x2, x3, x4, u = sp.symbols("x1, x2, x3, x4, u")
    xx = [x1, x2, x3, x4]

    # replace generalized coordinates with states
    if linearized:
        xu_subs = [(dq0_t, x3_t), (dq1_t, x4_t), (q0_t, x1_t), (q1_t, x2_t), (a, u_t)]
    else:
        xu_subs = [(dq0_t, x3_t), (dq1_t, x4_t), (q0_t, x1_t), (q1_t, x2_t), (tau, u_t)]


    # first order ODE (right hand side)
    dx_t = sp.Matrix([x3_t, x4_t, ddq[ddq0_t], ddq[ddq1_t]])
    dx_t = dx_t.subs(xu_subs)

    # linearized dynamics
    A = dx_t.jacobian(x_t)
    B = dx_t.diff(u_t)

    # symbolic expressions of A and B with parameter values
    Asym = A.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values)
    Bsym = B.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values)

    dx_t_sym = dx_t.subs(list(zip(x_t, xx))).subs(u_t, u).subs(params_values) # replacing all symbolic functions with symbols
    print(dx_t_sym)
    if linearized:
        lin = '_lin'
    else:
        lin = ''
    with open('c_files/acrobot'+ lin + '_ode.p', 'wb') as opened_file:
        pickle.dump(dx_t_sym, opened_file)
    with open('c_files/acrobot'+ lin + '_A.p', 'wb') as opened_file:
        pickle.dump(Asym, opened_file)
    with open('c_files/acrobot'+ lin + '_B.p', 'wb') as opened_file:
        pickle.dump(Bsym, opened_file)
    dxdt, A, B = load_existing()
    return dxdt, A, B

def load_existing(linearized=True):
    if linearized:
        lin = '_lin'
    else:
        lin = ''
    path = os.path.dirname(os.path.abspath(__file__))
    x1, x2, x3, x4, u = sp.symbols("x1, x2, x3, x4, u")
    with open(path + '/c_files/acrobot'+ lin + '_ode.p', 'rb') as opened_file:
        dx_t_sym = pickle.load(opened_file)
        print('Model loaded')
    with open(path+'/c_files/acrobot'+ lin + '_A.p', 'rb') as opened_file:
        Asym = pickle.load(opened_file)
        print('A matrix loaded')
    with open(path+'/c_files/acrobot'+ lin + '_B.p', 'rb') as opened_file:
        Bsym = pickle.load(opened_file)
        print('B matrix loaded')
    try:
        A_c_func = sp2c.convert_to_c((x1, x2, x3, x4, u), Asym,
                                     cfilepath=path+'/c_files/acrobot'+ lin + '_A.c',
                                     use_exisiting_so=False)
        B_c_func = sp2c.convert_to_c((x1, x2, x3, x4, u), Bsym,
                                     cfilepath=path+'/c_files/acrobot'+ lin + '_B.c',
                                     use_exisiting_so=False)
        A = lambda x, u: A_c_func(*x, *u)
        B = lambda x, u: B_c_func(*x, *u)
        dx_c_func = sp2c.convert_to_c((x1, x2, x3, x4, u), dx_t_sym, cfilepath=path + '/c_files/acrobot'+ lin + '_ode.c',
                                      use_exisiting_so=False)
        dxdt = lambda t, x, u: dx_c_func(*x, *u).T[0]
        assert (any(dxdt(0, [0, 0, 1., 1.], [0]) != [0., 0., 0., 0.]))
        print('Using C-function')
    except:
        A_func = sp.lambdify((x1, x2, x3, x4, u), Asym, modules="numpy")
        B_func = sp.lambdify((x1, x2, x3, x4, u), Bsym, modules="numpy")
        A = lambda x, u: A_func(*x, *u)
        B = lambda x, u: B_func(*x, *u)
        dx_func = sp.lambdify((x1, x2, x3, x4, u), dx_t_sym[:], modules="numpy")  # creating a callable python function
        dxdt = lambda t, x, u: np.array(dx_func(*x, *u))
        assert (any(dxdt(0, [0, 0, 1., 1.], [0]) != [0., 0., 0., 0.]))
        print('Using lambdify')
    return dxdt, A, B


if __name__ == "__main__":
    # execute only if run as a script
    modeling(linearized=False)