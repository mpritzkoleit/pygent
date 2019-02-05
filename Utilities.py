from numba import jit
import torch
from torch.autograd import grad
import numpy as np
import scipy as sci
import sympy as sp

def nth_derivative(f, wrt, n):
    # from: https://stackoverflow.com/questions/50322833/higher-order-gradients-in-pytorch
    # computes the n-th derivative of a function #not the hessian or the jacobian
    for i in range(n):

        grads = grad(f, wrt, create_graph=True, retain_graph=True)[0]
        f = grads.sum()
    return grads


def observation(x, xIsAngle):
    obsv = []
    for i, state in enumerate(x):
        if xIsAngle[i]:
            obsv.append(np.cos(state))
            obsv.append(np.sin(state))
        else:
            obsv.append(state)

    return obsv


def torch_jacobian(f, x):
    # computes the jacobian matrix
    fx = nth_derivative(f, x, n=1)
    fxx = nth_derivative(fx.narrow(0, 0, 1), x, n=1).unsqueeze(0)
    for i in range(1, max(x.shape)):
        fxx = torch.cat((fxx, nth_derivative(fx.narrow(0, i, 1), x, n=1).unsqueeze(0)))
    return fxx


def hessian(f, xx, uu, eps=1e-3):
    # finite differences approximation of d^2f(x, u)/dxdu
    # http://www.iue.tuwien.ac.at/phd/heinzl/node27.html Eq. 2.52
    xDim = len(xx)
    uDim = len(uu)
    H = np.zeros([xDim, uDim])
    for i, x in enumerate(xx):
        for j, u in enumerate(uu):
            ex = np.eye(1, xDim, i)[0]*eps
            eu = np.eye(1, uDim, j)[0]*eps
            H[i, j] = (f(xx+ex, uu+eu) - f(xx-ex, uu+eu) - f(xx+ex, uu-eu) + f(xx-ex, uu-eu))/(4*eps**2)
    return H

def hessian2(f, xx, uu, eps=1e-3):
    # finite differences approximation of d^2f(x, u)/dxdu
    # http://www.iue.tuwien.ac.at/phd/heinzl/node27.html Eq. 2.52
    xDim = len(xx)
    uDim = len(uu)
    H = np.zeros([xDim, xDim])
    for i, xi in enumerate(xx):
        for j, xj in enumerate(xx):
            exi = np.eye(1, xDim, i)[0]*eps
            exj = np.eye(1, xDim, j)[0]*eps
            H[i, j] = (f(xx+exi+exj, uu) - f(xx-exi+exj, uu) - f(xx+exi-exj, uu) + f(xx-exi-exj, uu))/(4*eps**2)
    return H

def hessian3(f, xx, uu, eps=1e-3):
    # finite differences approximation of d^2f(x, u)/dxdu
    # http://www.iue.tuwien.ac.at/phd/heinzl/node27.html Eq. 2.52
    xDim = len(xx)
    uDim = len(uu)
    H = np.zeros([uDim, uDim])
    for i, ui in enumerate(uu):
        for j, uj in enumerate(uu):
            eui = np.eye(1, uDim, i)[0]*eps
            euj = np.eye(1, uDim, j)[0]*eps
            H[i, j] = (f(xx, uu+eui+euj) - f(xx, uu-eui+euj) - f(xx, uu+eui-euj) + f(xx, uu-eui-euj))/(4*eps**2)
    return H

def system_linearization(f, xx, uu, eps=1e-3):
    # finite differences approximation of A = df/dx and B = df/du
    # http://www.iue.tuwien.ac.at/phd/heinzl/node27.html Eq. 2.52
    xDim = len(xx)
    uDim = len(uu)
    if np.isscalar(f(xx, uu)):
        outDim = 1
    else:
        outDim = len(f(xx, uu))

    A = np.zeros([outDim, xDim])
    B = np.zeros([outDim, uDim])
    for i, x in enumerate(xx):
        ex = np.eye(1, xDim, i)[0] * eps
        A[:, i] = (f(xx+ex, uu) - f(xx-ex, uu))/(2*eps)
    for j, u in enumerate(uu):
        eu = np.eye(1, uDim, j)[0]*eps
        B[:, j] = (f(xx, uu+eu) - f(xx, uu-eu))/(2*eps)
    return A, B

def unpackBlockMatrix(A, n, m):
    Ann = A[:n, :n]
    Anm = A[:n, n:]
    Amn = A[n:, :n]
    Amm = A[n:, n:]

    return Ann, Anm, Amn, Amm


def c2d(A, B, dt):
    # https://en.wikipedia.org/wiki/Discretization
    n = len(A)
    m = len(B[0, :])
    M = np.block([[A, B], [np.zeros([m, n]), np.zeros([m, m])]])
    Md = sci.linalg.expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

