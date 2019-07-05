import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import time
import copy
import inspect
from shutil import copyfile
import sympy as sp
try:
    from sympy_to_c import sympy_to_c as sp2c
except ImportError:
    print('sympy-to-c could not be imported!')
import os
import cvxopt as opt
opt.solvers.options['show_progress'] = False

# pygent
from pygent.helpers import c2d, system_linearization
from pygent.agents import FeedBack, Agent
from pygent.algorithms.core import Algorithm
from pygent.helpers import mapAngles


class iLQR(Algorithm):
    """ iLQR - iterative linear-quadratic regulator with box control constraints.

    Implementation based on:

    https://de.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization

    Papers:

    1) Y. Tassa, T. Erez, E. Todorov: Synthesis and Stabilization of Complex Behaviours through
    Online Trajectory Optimization
    Link: https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf

    2) Y. Tassa, N. Monsard, E. Todorov: Control-Limited Differential Dynamic Programming
    Link: https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

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
    def __init__(self, environment, t, dt,
                 maxIters=500,
                 tolGrad=1e-4,
                 tolFun=1e-7,
                 fastForward=False,
                 path='../results/ilqr/',
                 fcost=None,
                 constrained=False,
                 save_interval=10,
                 printing=True):
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
            fcost (function): Final cost function. c = fcost(x_N)
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
        if not os.path.isdir(path + 'c_files/'):
            os.makedirs(path + 'c_files/')
        copyfile(inspect.stack()[-1][1], path + 'exec_script.py')
        self.fastForward = fastForward  # if True use Eulers Method instead of ODE solver
        agent = FeedBack(None, self.uDim) 
        super(iLQR, self).__init__(environment, agent, t, dt)
        self.environment.xIsAngle = np.zeros(self.xDim, dtype=bool)
        self.xIsAngle = self.environment.xIsAngle
        self.cost = 0.
        if fcost == None:
            self.fcost_fnc = lambda x, mod: self.cost_fnc(x, np.zeros((1, self.uDim)), mod)
        else:
            if inspect.signature(fcost).parameters.__len__() == 1:
                self.fcost_fnc = lambda x, mod: fcost(x)  # final cost
        self.cost_init()
        self.init_trajectory()
        self.current_alpha = 1
        self.printing = printing
        # todo: mu to eta

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
        self.save_interval = save_interval
        self.parallel = False

        self.KK = []
        self.kk = []

    def cost_fnc(self, x, u, mod):
        """

        Args:
            x:
            u:

        Returns:

        """
        c = self.environment.cost(x, u, None, mod)*self.dt
        return c


    def forward_pass(self, alpha):
        """

        Args:
            alpha: line search parameter

        Returns:

        """
        xx_ = self.xx
        uu_ = self.uu

        self.environment.reset(self.environment.x0)
        self.agent.reset()

        cost = 0
        for i in range(self.steps):
            u = self.KK[i] @ (self.environment.x - xx_[i]) + alpha*self.kk[i].T[0] + uu_[i] # eq. (7b)

            if self.constrained:
                u = np.clip(u, -self.environment.uMax, self.environment.uMax)
            self.agent.control(self.dt, u)

            if self.fastForward:
                c = self.environment.fast_step(u, self.dt)
            else:
                c = self.environment.step(u, self.dt)
            cost += c

        cost += self.fcost_fnc(self.environment.x, np)*self.dt

        xx = self.environment.history
        uu = self.agent.history[1:]

        

        return xx, uu, cost, self.environment.terminated

    def backward_pass(self):
        x = self.xx[-1]
        system_matrices = [self.linearization(x, u) for x, u in zip(self.xx[:-1], self.uu)]
        cost_matrices = [self.cost_lin(x, u) for x, u in zip(self.xx[:-1], self.uu)]

        # DARE in Vt, vt
        vx = self.cfx(x)*self.dt
        Vxx = self.Cfxx(x)*self.dt

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
            Cxx, Cuu, Cxu, cx, cu = cost_matrices[i]

            # expanded system dynamics
            Ad, Bd, ft = system_matrices[i]

            # eq. (5a,5b), paper 1)
            qx = cx + np.matmul(Ad.T, vx)
            qu = cu + np.matmul(Bd.T, vx)

            # eq (5c-e), paper 1)
            Qxx = Cxx + np.matmul(Ad.T, np.matmul(Vxx, Ad))
            Quu = Cuu + np.matmul(Bd.T, np.matmul(Vxx, Bd))
            Qux = Cxu.T + np.matmul(Bd.T, np.matmul(Vxx, Ad))

            VxxReg = Vxx + self.mu * np.eye(self.xDim)*(self.regType == 1)

            # eq. (10a,10b), paper 1)
            QuuReg = Cuu + np.matmul(Bd.T, np.matmul(VxxReg, Bd)) + self.mu * np.eye(self.uDim) * (self.regType == 2)
            QuxReg = Cxu.T + np.matmul(Bd.T, np.matmul(VxxReg, Ad))

            try:
                np.linalg.cholesky(QuuReg)
            except np.linalg.LinAlgError as e:
                print('Quu not positive-definite')
                success = False
                break

            if self.constrained:  # solve QP, eq. (11), paper 2)
                # convert matrices
                QuuOpt = opt.matrix(QuuReg)
                quOpt = opt.matrix(qu)

                # inequality constraints Gx <= h, where x is the decision variable
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
                    Kt[free_controls, :] = -np.linalg.solve(QuuReg, QuxReg)[free_controls,:]

            else: # analytic solution
                kt = -np.linalg.solve(QuuReg, qu) # eq. (10c), paper 1)
                Kt = -np.linalg.solve(QuuReg, QuxReg) # eq. (10b), paper 1)


            vx = qx + Kt.T@Quu@kt + Kt.T@qu + Qux.T@kt
            Vxx = Qxx + Kt.T@Quu@Kt + Kt.T@Qux + Qux.T@Kt
            Vxx = 0.5*(Vxx + Vxx.T)  # remain symmetry

            dV2 = 0.5*kt.T@Quu@kt
            dV1 = kt.T@qu

            # save Kt, kt
            self.KK.insert(0, Kt)
            self.kk.insert(0, kt)

            V1.insert(0, dV1)
            V2.insert(0, dV2)

        return V1, V2, success

    def run(self, x0):
        self.environment.reset(x0)
        self.xx, self.uu, self.cost, terminated = self.forward_pass(1.)
        pass

    def run_disk(self, x0):
        self.environment.reset(x0)
        self.KK = np.load(self.path + 'data/K_.npy')
        self.kk = np.load(self.path + 'data/k.npy')
        self.uu = np.load(self.path + 'data/uu.npy')
        self.xx = np.load(self.path + 'data/xx.npy')
        self.current_alpha = np.load(self.path + 'data/alpha.npy')
        print(self.current_alpha)
        self.xx, self.uu, self.cost, terminated = self.forward_pass(self.current_alpha)
        pass


    def run_optim(self):
        """ Trajectory optimization. """
        start_time = time.time()
        self.mu = 1.0
        success_gradient = False
        # later run_episode
        for _ in range(1, self.max_iters + 1):
            success_bw = False
            while not success_bw:
                V1, V2, success_bw = self.backward_pass()
                if not success_bw:
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
                # todo: add parallel linesearch
                x_list = []
                u_list = []
                cost_list = []
                for a_index, alpha in enumerate(self.alphas):
                    #print('Linesearch:', a_index+1, '/', len(self.alphas))
                    xx, uu, cost, sys_terminated = self.forward_pass(alpha)
                    x_list.append(xx)
                    u_list.append(uu)
                    cost_list.append(cost)
                    if np.any(xx > 1e8):# or sys_terminated:
                        print('forward diverged.')
                        break
                    # cost difference between iterations
                    dcost = self.cost - cost

                    # expected cost reduction, paper 1)
                    expected = -alpha * (np.sum(V1) + alpha * np.sum(V2))

                    # check if expected cost is > 0
                    if expected > 0.:
                        z = dcost / expected
                    else:
                        z = np.sign(dcost)
                        print('non-positive expected reduction')
                        #self.increase_mu() # todo: probably delete this line, if something's not working!
                    if z > self.zmin:
                        success_fw = True
                        if not self.parallel:
                            break
            if success_fw:
                #todo: cost instead of self.cost in print?
                if self.printing:
                    print('Iter. ', _, '| Cost: %.7f' % cost, ' | red.: %.5f' % dcost, '| exp.: %.5f' % expected)
                best_idx = np.argmin(cost_list)
                self.cost = np.copy(cost_list[best_idx])
                self.xx = np.copy(x_list[best_idx])
                self.uu = np.copy(u_list[best_idx])
                self.current_alpha = self.alphas[best_idx]

                # decrease mu
                self.decrease_mu()

                if dcost < self.tolFun:
                    if self.printing:
                        print('Converged: small improvement')
                    break
                if success_gradient:
                    if self.printing:
                        print('Converged: small gradient')
                    break
                if _ % self.save_interval == 0:
                    self.save()
            else:
                self.increase_mu()
                if self.printing:
                    print('Iter. ', _, '| Forward not successfull')
                if self.mu > self.mu_max:
                    if self.printing:
                        print('Diverged: no improvement')
                    break

        print('Iterations: %d | Final Cost-to-Go: %.2f | Runtime: %.2f min.' % (_, self.cost, (time.time() - start_time) / 60))
        self.save() # save controller
        return self.xx, self.uu, self.cost

    def init_trajectory(self):
        """ Initial trajectory, with u=0. """

        for _ in range(self.steps):
            u = np.random.uniform(0.001, 0.001, self.uDim)
            u = self.agent.control(self.dt, u)
            # necessary to store control in agents history
            if self.fastForward:
                c = self.environment.fast_step(u, self.dt)
            else:
                c = self.environment.step(u, self.dt)
            self.cost += c
        self.cost += self.fcost_fnc(self.environment.x, np) * self.dt
        self.xx = self.environment.history
        self.uu = self.agent.history[1:]
        pass

    def linearization(self, x, u):
        """ Computes the 1st order expansion of the system dynamics in discrete form.

        Args:
            """
        A, B = system_linearization(lambda xx, uu: self.environment.ode(None, xx, uu), x, u)

        # Ad, Bd = c2d(A, B, self.dt)
        Ad = A*self.dt + np.eye(self.xDim)
        Bd = B*self.dt
        fd = np.zeros((self.xDim, 1))
        return Ad, Bd, fd

    def cost_init(self):
        """ Computes second order expansion of the """
        # 2nd order taylor expansion of the cost function along a trajectory

        xx = sp.symbols('x1:' + str(self.xDim + 1))
        uu = sp.symbols('u1:' + str(self.uDim + 1))

        c = self.cost_fnc(xx, uu, sp)
        cc = sp.Matrix([[c]])
        cx = cc.jacobian(xx)
        cu = cc.jacobian(uu)
        Cxx = cx.jacobian(xx)
        Cuu = cu.jacobian(uu)
        Cxu = cx.jacobian(uu)

        # final cost
        cf = self.fcost_fnc(xx, sp)
        ccf = sp.Matrix([[cf]])
        cfx = ccf.jacobian(xx)
        Cfxx = cfx.jacobian(xx)

        try:
            cx_func = sp2c.convert_to_c((*xx, *uu), cx.T, cfilepath=self.path + 'c_files/cx.c', use_exisiting_so=False)
            self.cx = lambda x, u: cx_func(*x, *u)
            cu_func = sp2c.convert_to_c((*xx, *uu), cu.T, cfilepath=self.path + 'c_files/cu.c', use_exisiting_so=False)
            self.cu = lambda x, u: cu_func(*x, *u)
            Cxx_func = sp2c.convert_to_c((*xx, *uu), Cxx, cfilepath=self.path + 'c_files/Cxx.c', use_exisiting_so=False)
            self.Cxx = lambda x, u: Cxx_func(*x, *u)
            Cuu_func = sp2c.convert_to_c((*xx, *uu), Cuu, cfilepath=self.path + 'c_files/Cuu.c', use_exisiting_so=False)
            self.Cuu = lambda x, u: Cuu_func(*x, *u)
            Cxu_func = sp2c.convert_to_c((*xx, *uu), Cxu, cfilepath=self.path + 'c_files/Cxu.c', use_exisiting_so=False)
            self.Cxu = lambda x, u: Cxu_func(*x, *u)
            cfx_func = sp2c.convert_to_c((*xx,), cfx.T, cfilepath=self.path + 'c_files/cfx.c', use_exisiting_so=False)
            self.cfx = lambda x: cfx_func(*x,)
            Cfxx_func = sp2c.convert_to_c((*xx,), Cfxx, cfilepath=self.path + 'c_files/Cfxx.c', use_exisiting_so=False)
            self.Cfxx = lambda x: Cfxx_func(*x,)
            print('Using C functions for taylor expansion of the cost function!')
        except:
            print('Could not use C functions for taylor expansion of the cost function!')
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
        x = mapAngles(self.xIsAngle, x)
        Cxx = self.Cxx(x, u)
        Cuu = self.Cuu(x, u)
        Cxu = self.Cxu(x, u)
        cx = self.cx(x, u)
        cu = self.cu(x, u)
        return Cxx, Cuu, Cxu, cx, cu

    def decrease_mu(self):
        """ Decrease regularization parameter mu. Section F, paper 1)"""
        self.mu_d = min(self.mu_d / self.mu_d0, 1 / self.mu_d0)
        self.mu = self.mu * self.mu_d * (self.mu > self.mu_min)
        pass

    def increase_mu(self):
        """ Increase regularization parameter mu. Section F, paper 1)"""
        self.mu_d = max(self.mu_d0, self.mu_d0 * self.mu_d)
        self.mu = max(self.mu_min, self.mu * self.mu_d)
        pass

    def plot(self):
        self.environment.plot()
        self.environment.save_history('environment', self.path + 'data/')
        plt.savefig(self.path + 'plots/environment.pdf')
        self.agent.plot()
        self.agent.save_history('agent', self.path + 'data/')
        plt.savefig(self.path + 'plots/controller.pdf')
        plt.close('all')

    def animation(self):
        ani = self.environment.animation()
        if ani != None:
            try:
                ani.save(self.path + 'animations/animation.mp4', fps=1 / self.dt)
            except:
                ani.save(self.path + 'animations/animation.gif', fps=1 / self.dt)
        plt.close('all')

    def save(self):
        np.save(self.path + 'data/K_', self.KK)
        np.save(self.path + 'data/k', self.kk)
        np.save(self.path + 'data/uu', self.uu)
        np.save(self.path + 'data/xx', self.xx)
        np.save(self.path + 'data/alpha', self.current_alpha)
        if self.printing:
            self.plot()