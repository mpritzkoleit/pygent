from numba import jit
import numpy as np
from Agents import FeedBack
from Algorithms import Algorithm
import matplotlib.pyplot as plt
import time
import copy
from Utilities import c2d, system_linearization
import sympy as sp

try:
    from sympy_to_c import sympy_to_c as sp2c
except ImportError:
    print('sympy-to-c could not be imported!')
import os
import cvxopt as opt

opt.solvers.options['show_progress'] = False


class iLQR(Algorithm):
    """ iLQR - iterative linear-quadratic regulator with box control constraints.

    Implementation based on:

    # MATLAB Yuval Tassa

    Attributes:
        n (int): number of episodes
        t (int, float): episode length
        dt (int, float): step size
        meanCost (array): mean cost of an episode
        agent (Agent(object)): agent of the algorithm
        environment (Environment(object)): environment
        eps (float [0, 1]): epsilon-greedy action(control)
        nData: maximum length of data set

    """

    def __init__(self, environment, t, dt, maxIters=500, tolGrad=1e-4,
                 tolFun=1e-7, fastForward=False, path='../Results/iLQR/', fcost=None, constrained=False):
        """

        Args:
            environment (Environment):
            t (float): optimization time
            dt (float): step-size
            maxIters (int): maximum number of iterations
            tolGrad (float):
            tolFun (float):
            fastForward (bool): if True, use Euler forward integration
            path (string): directory for saving time-varying controllers and trajectories
            fcost (callable): Final cost function. c = fcost(x_N)
            constrained (bool): if True, control input is constrained
        """

        self.uDim = environment.uDim
        self.xDim = environment.xDim
        self.path = path
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path + 'plots/'):
            os.makedirs(path + 'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')
        self.fastForward = fastForward  # if True use Eulers Method instead of ODE solver
        agent = FeedBack(None, self.uDim)
        super(iLQR, self).__init__(environment, agent, t, dt)
        self.environment.xIsAngle = np.zeros(self.xDim, dtype=bool)
        self.cost = 0.
        self.fcost_fnc = fcost  # final cost
        if fcost == None:
            self.fcost_fnc = lambda x: self.cost_fnc(x, np.zeros((1, self.uDim)))

        self.cost_init()
        self.init_trajectory()

        # algorithm parameters
        self.max_iters = maxIters
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.mu_d0 = 1.6
        self.mu_d = 1.
        self.mu = 1.
        self.alphas = 10 ** np.linspace(0, -3, 11)
        self.zmin = 0.
        self.tolGrad = tolGrad
        self.tolFun = tolFun
        self.constrained = constrained
        self.lims = self.environment.uMax
        self.regType = 2.

    def cost_fnc(self, x, u):
        """

        Args:
            x:
            u:

        Returns:

        """
        c = self.environment.cost(x, u, None)*self.dt
        return c


    def forward_pass(self, alpha):
        """

        Args:
            alpha:

        Returns:

        """
        xx_ = self.xx
        uu_ = self.uu

        self.environment.reset(self.environment.x0)
        self.agent.reset()

        cost = 0
        for i in range(self.steps):
            u = self.KK[i] @ (self.environment.x - xx_[i]) + alpha * self.kk[i].T[0] + uu_[i]
            if self.constrained:
                u = np.clip(u, -self.environment.uMax, self.environment.uMax)
            self.agent.control(self.dt, u)

            if self.fastForward:
                c = self.environment.fast_step(self.dt, u)
            else:
                c = self.environment.step(self.dt, u)
            cost += c
        cost += self.fcost_fnc(self.environment.x)*self.dt
        xx = self.environment.history
        uu = self.agent.history[1:]

        return xx, uu, cost

    def backward_pass(self):
        x = self.xx[-1]
        system_matrices = [self.linearization(x, u) for x, u in zip(self.xx[:-1], self.uu)]
        cost_matrices = [self.cost_lin(x, u) for x, u in zip(self.xx[:-1], self.uu)]
        # DARE in Vt, vt
        vx = self.cfx(x) * self.dt
        Vxx = self.Cfxx(x) * self.dt

        dV1 = np.zeros((1, 1))
        dV2 = np.zeros((1, 1))

        self.KK = []
        self.kk = []

        V1 = [dV1]
        V2 = [dV2]

        success = True

        for i in range(self.steps - 1, -1, -1):
            x = self.xx[i]
            u = self.uu[i]

            # expanded cost
            Cxx, Cuu, Cxu, cx, cu = cost_matrices[i]  # self.cost_lin(x, u)

            # expanded system dynamics
            Ad, Bd, ft = system_matrices[i]
            # eq. 5a/b
            qx = cx + np.matmul(Ad.T, vx)  # + np.matmul(Ad.T, np.matmul(Vt, ft))
            qu = cu + np.matmul(Bd.T, vx)  # +np.matmul(Bd.T, np.matmul(Vt, ft))

            # eq 5c-e
            Qxx = Cxx + np.matmul(Ad.T, np.matmul(Vxx, Ad))
            Quu = Cuu + np.matmul(Bd.T, np.matmul(Vxx, Bd))
            Qux = Cxu.T + np.matmul(Bd.T, np.matmul(Vxx, Ad))

            VxxReg = Vxx + self.mu * np.eye(self.xDim) * (self.regType == 1)
            QuxReg = Cxu.T + np.matmul(Bd.T, np.matmul(VxxReg, Ad))
            QuuReg = Cuu + np.matmul(Bd.T, np.matmul(VxxReg, Bd)) + self.mu * np.eye(self.uDim) * (self.regType == 2)

            if self.constrained:  # solve QP
                QuuOpt = opt.matrix(QuuReg)
                quOpt = opt.matrix(qu)
                G = np.kron(np.array([[1.], [-1.]]), np.eye(self.uDim))
                h = np.array([self.lims - u, self.lims + u]).reshape((2 * self.uDim,))
                GOpt = opt.matrix(G)
                hOpt = opt.matrix(h)
                sol = opt.solvers.qp(QuuOpt, quOpt, GOpt, hOpt)
                kt = np.array(sol['x'])
                clamped = np.zeros((self.uDim), dtype=bool)
                # adapted from boxQP.m of iLQG package
                uplims = np.isclose(kt.reshape(self.uDim, ), h[0:self.uDim], atol=1e-3)
                lowlims = np.isclose(kt.reshape(self.uDim, ), h[self.uDim::], atol=1e-3)

                clamped[uplims] = True
                clamped[lowlims] = True
                free_controls = np.logical_not(clamped)
                Kt = np.zeros((self.uDim, self.xDim))
                if any(free_controls):
                    Kt[free_controls, :] = -np.linalg.solve(QuuReg, QuxReg)[free_controls,
                                            :]  # -np.matmul(QuuInv, QuxReg)
            else:
                try:
                    np.linalg.cholesky(QuuReg)

                except np.linalg.LinAlgError as e:
                    print('Quu not positive-semidefinite')
                    success = False
                    break

                kt = -np.linalg.solve(QuuReg, qu)  # -np.matmul(QuuInv, qu)#
                Kt = -np.linalg.solve(QuuReg, QuxReg)  # -np.matmul(QuuInv, Qux)

            vx = qx + Kt.T@Quu@kt + Kt.T@qu + Qux.T@kt
            Vxx = Qxx + Kt.T@Quu@Kt + Kt.T@Qux + Qux.T@Kt
            Vxx = 0.5 * (Vxx + Vxx.T)  # remain symmetry

            # dV2 = 0.5 * np.matmul(kt.T, np.matmul(QuuReg, kt))
            # dV1 = np.matmul(kt.T, qu)
            dV2 = 0.5 * kt.T @ Quu @ kt
            dV1 = kt.T @ qu
            # save Kt, kt
            self.KK.insert(0, Kt)
            self.kk.insert(0, kt)

            V1.insert(0, dV1)
            V2.insert(0, dV2)

        return V1, V2, success

    def run(self, x0):
        self.environment.reset(x0)
        self.KK = np.load(self.path + 'data/K_.npy')
        self.kk = np.load(self.path + 'data/k.npy')
        self.uu = np.load(self.path + 'data/uu.npy')
        self.xx = np.load(self.path + 'data/xx.npy')
        self.xx, self.uu, self.cost = self.forward_pass(1.)
        pass


    def run_optim(self):
        start_time = time.time()
        self.mu = 1.0
        success_gradient = False
        # later run_episode
        for _ in range(1, self.max_iters + 1):
            success_bw = False
            while not success_bw:
                V1, V2, success_bw = self.backward_pass()
                if not success_bw:
                    # print('Backward successfull')
                    print('diverged')
                    # increase mu
                    self.increase_mu()
                    break

            success_fw = False
            if success_bw:
                # check for gradient
                g_norm = np.mean(
                    np.max(np.abs(np.array(self.kk).reshape((len(self.kk), self.uDim)) / (np.abs(self.uu) + 1)),
                           axis=0))
                if g_norm < self.tolGrad and self.mu < 1e-5:
                    self.decrease_mu()
                    success_gradient = True

                # Line-search
                for a_index, alpha in enumerate(self.alphas):
                    print('Linesearch:', a_index, '/', len(self.alphas))
                    xx, uu, cost = self.forward_pass(alpha)
                    if np.any(xx > 1e8):
                        print('forward diverged.')
                        break
                    dcost = self.cost - cost
                    expected = -alpha * (np.sum(V1) + alpha * np.sum(V2))
                    if expected > 0.:
                        z = dcost / expected
                    else:
                        z = np.sign(dcost)
                        print('non-positive expected reduction')
                    # print('z :', z)
                    if z > self.zmin:
                        success_fw = True
                        # print('Forward successfull')
                        break
                    # print('Forward not successfull')
            if success_fw:
                print('Iter. ', _, '| Cost: %.7f' % self.cost, ' | red.: %.5f' % dcost, '| exp.: %.5f' % expected)
                self.cost = np.copy(cost)
                self.xx = np.copy(xx)
                self.uu = np.copy(uu)

                # decrease mu
                self.decrease_mu()

                if dcost < self.tolFun:
                    print('converged: small improvement')
                    break
                if success_gradient:
                    print('converged: small gradient')
                    break
            else:
                self.increase_mu()
                print('Iter. ', _, '| Forward not successfull')
                if self.mu > self.mu_max:
                    print('converged: no improvement')
                    break
        print('Iterations: %d | Final Cost-to-Go: %.2f | Runtime: %.2f min.' % (
        _, self.cost, (time.time() - start_time) / 60))
        self.save()
        return self.xx, self.uu, self.cost

    def init_trajectory(self):
        # random trajectory
        for _ in range(self.steps):
            u = np.random.uniform(0., 0., self.uDim)
            u = self.agent.control(self.dt, u)
            # necessary to store control in agents history
            if self.fastForward:
                c = self.environment.fast_step(self.dt, u)
            else:
                c = self.environment.step(self.dt, u)
            self.cost += c
        self.cost += self.fcost_fnc(self.environment.x) * self.dt
        self.xx = self.environment.history
        self.uu = self.agent.history[1:]
        pass

    def linearization(self, x, u):

        A, B = system_linearization(lambda xx, uu: self.environment.ode(None, xx, uu), x, u)

        # Ad, Bd = c2d(A, B, self.dt)
        Ad = A * self.dt + np.eye(self.xDim)
        Bd = B * self.dt
        # Ft = np.block([[self.Ad(x, u), self.Bd(x, u)]])
        # Ft = np.block([[Ad, Bd]])
        fd = np.zeros((self.xDim, 1))  # self.dt *0* np.expand_dims(self.environment.ode(None, x, u), 0).T
        return Ad, Bd, fd

    def cost_init(self):
        # 2nd order taylor expansion of the cost function along a trajectory
        # self.environment.

        xx = sp.symbols('x1:' + str(self.xDim + 1))
        uu = sp.symbols('u1:' + str(self.uDim + 1))

        c = self.cost_fnc(xx, uu)
        cc = sp.Matrix([[c]])
        cx = cc.jacobian(xx)
        cu = cc.jacobian(uu)
        Cxx = cx.jacobian(xx)
        Cuu = cu.jacobian(uu)
        Cxu = cx.jacobian(uu)

        # final cost
        cf = self.fcost_fnc(xx)
        ccf = sp.Matrix([[cf]])
        cfx = ccf.jacobian(xx)
        Cfxx = cfx.jacobian(xx)

        try:
            assert(False)
            cx_func = sp2c.convert_to_c((*xx, *uu), cx.T, cfilepath="cx.c", use_exisiting_so=False)
            self.cx = lambda x, u: cx_func(*x, *u)
            cu_func = sp2c.convert_to_c((*xx, *uu), cu.T, cfilepath="cu.c", use_exisiting_so=False)
            self.cu = lambda x, u: cu_func(*x, *u)
            Cxx_func = sp2c.convert_to_c((*xx, *uu), Cxx, cfilepath="Cxx.c", use_exisiting_so=False)
            self.Cxx = lambda x, u: Cxx_func(*x, *u)
            Cuu_func = sp2c.convert_to_c((*xx, *uu), Cuu, cfilepath="Cuu.c", use_exisiting_so=False)
            self.Cuu = lambda x, u: Cuu_func(*x, *u)
            Cxu_func = sp2c.convert_to_c((*xx, *uu), Cxu, cfilepath="Cxu.c", use_exisiting_so=False)
            self.Cxu = lambda x, u: Cxu_func(*x, *u)
            cfx_func = sp2c.convert_to_c((*xx,), cfx.T, cfilepath="cfx.c", use_exisiting_so=False)
            self.cfx = lambda x: cfx_func(*x,)
            Cfxx_func = sp2c.convert_to_c((*xx,), Cfxx, cfilepath="Cfxx.c", use_exisiting_so=False)
            self.Cfxx = lambda x: Cfxx_func(*x,)
        except:
            print('Could not use sympy-to-c cost functions!')
            self.cx = sp.lambdify((xx, uu), cx.T)
            self.cu = sp.lambdify((xx, uu), cu.T)
            self.Cxx = sp.lambdify((xx, uu), Cxx)
            self.Cuu = sp.lambdify((xx, uu), Cuu)
            self.Cxu = sp.lambdify((xx, uu), Cxu)
            self.cfx = sp.lambdify((xx,), cfx.T)
            self.Cfxx = sp.lambdify((xx,), Cfxx)

        pass

    def sys_init(self):
        xx = sp.symbols('x1:' + str(self.xDim + 1))
        uu = sp.symbols('u1:' + str(self.uDim + 1))
        dx = self.environment.ode(None, xx, uu)
        dx = sp.Matrix([[dx]]).T
        A = dx.jacobian(xx)
        B = dx.jacobian(uu)
        Ad, Bd = c2d(A, B, self.dt)
        self.Ad = sp.lambdify((xx, uu), Ad)
        self.Bd = sp.lambdify((xx, uu), Bd)
        pass

    def cost_lin(self, x, u):
        Cxx = self.Cxx(x, u)
        Cuu = self.Cuu(x, u)
        Cxu = self.Cxu(x, u)
        cx = self.cx(x, u)
        cu = self.cu(x, u)
        return Cxx, Cuu, Cxu, cx, cu

    def decrease_mu(self):
        self.mu_d = min(self.mu_d / self.mu_d0, 1 / self.mu_d0)
        self.mu = self.mu * self.mu_d * (self.mu > self.mu_min)
        pass

    def increase_mu(self):
        self.mu_d = max(self.mu_d0, self.mu_d0 * self.mu_d)
        self.mu = max(self.mu_min, self.mu * self.mu_d)
        pass

    def plot(self):
        self.environment.plot()
        plt.savefig(self.path + 'plots/environment.pdf')
        plt.savefig(self.path + 'plots/environment.pgf')
        self.agent.plot()
        plt.savefig(self.path + 'plots/controller.pdf')
        plt.savefig(self.path + 'plots/controller.pgf')
        plt.close('all')

    def animation(self):
        ani = self.environment.animation()
        if ani != None:
            ani.save(self.path + 'animations/animation.mp4', fps=1 / self.dt)
        plt.close('all')

    def save(self):
        np.save(self.path + 'data/K_', self.KK)
        np.save(self.path + 'data/k', self.kk)
        np.save(self.path + 'data/uu', self.uu)
        np.save(self.path + 'data/xx', self.xx)


class NMPC(iLQR):
    """ Nonlinear model predictive control (NMPC) algorithm based on iLQR.

    Args:
        horizon (int): optimization horizon. """

    def __init__(self, environment, t, dt, horizon=100, maxIters=1, tolGrad=1e-4,
                 tolFun=1e-7, fastForward=True, path='../Results/iLQR/', constrained=False):
        self.sim_environment = copy.deepcopy(environment)
        self.sim_agent = FeedBack(None, self.sim_environment.uDim)
        self.horizon = horizon
        self.tsim = t
        self.sim_steps = int(t/dt)
        super(NMPC, self).__init__(environment=environment, t=horizon*dt, dt=dt, maxIters=maxIters, tolGrad=tolGrad,
                 tolFun=tolFun, fastForward=fastForward, path=path, fcost=None, constrained=constrained)

    def mpc_step(self):
        self.run_optim()
        u = self.sim_agent.control(self.dt, self.uu[0])
        print('u', u)
        c = self.sim_environment.step(self.dt, u)
        u0 = self.agent.control(self.dt, np.zeros(self.uDim))
        self.environment.step(self.dt, u0)
        self.xx = self.environment.history[1:]
        self.uu = self.agent.history[2:]
        self.environment.history = self.xx
        self.agent.history = self.uu
        self.environment.x0 = self.sim_environment.x
        return c

    def run_mpc(self):
        cost = 0
        for _ in range(self.sim_steps):
            print('NMPC step:', _, '/', self.sim_steps)
            cost += self.mpc_step()

    def plot(self):
        self.sim_environment.plot()
        plt.savefig(self.path + 'plots/environment.pdf')
        plt.savefig(self.path + 'plots/environment.pgf')
        self.sim_agent.plot()
        plt.savefig(self.path + 'plots/controller.pdf')
        plt.savefig(self.path + 'plots/controller.pgf')
        plt.close('all')

    def animation(self):
        ani = self.sim_environment.animation()
        if ani != None:
            ani.save(self.path + 'animations/animation.mp4', fps=1 / self.dt)
        plt.close('all')
