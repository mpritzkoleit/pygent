from numba import jit
import numpy as np
from Agents import FeedBack
from Data import DataSet
from Algorithms import Algorithm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import time
from Utilities import nth_derivative, torch_jacobian, hessian, hessian2, hessian3, c2d, unpackBlockMatrix, system_linearization
import sympy as sp

class iLQR(Algorithm):
    """ iLQR -

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

    def __init__(self, environment, t, dt, maxIters=500, tolGrad=1e-4, tolFun=1e-7, fastForward=False):
        self.uDim = environment.uDim
        self.xDim = environment.xDim
        self.fastForward = fastForward  # if True use Eulers Method instead of ODE solver
        agent = FeedBack(None, self.uDim)
        super(iLQR, self).__init__(environment, agent, t, dt)
        self.environment.xIsAngle = np.zeros(self.xDim, dtype=bool)
        self.cost = 0.
        #self.sys_init()
        self.cost_init()
        self.init_trajectory()
        self.finalcost = 10. # c(x_N) = finalcost*c(x_k, None)
        self.max_iters = maxIters
        self.mu_min = 1e-6
        self.mu_max = 1e6
        self.mu_d0 = 1.5
        self.mu_d = 1.
        self.mu = 1.0
        self.alphas = 10**np.linspace(0, -3, 11)
        self.zmin = 0.
        self.tolGrad = tolGrad
        self.tolFun = tolFun

    def cost_fnc(self, x, u):
        c = self.environment.cost(x, u, None)
        return c


    def forward_pass(self, alpha):
        xx_ = self.xx
        uu_ = self.uu

        self.environment.reset(self.environment.x0)
        self.agent.reset()

        cost = 0
        for i in range(self.steps):
            u = (np.matmul(self.KK[i], (self.environment.x - xx_[i])) + alpha * self.kk[i] + uu_[i])[0]

            self.agent.control(self.dt, u)

            if self.fastForward:
                c = self.environment.fast_step(self.dt, u)
            else:
                c = self.environment.step(self.dt, u)

            cost += c*self.dt

        xx = self.environment.history
        uu = self.agent.history[1:]

        return xx, uu, cost


    def backward_pass(self):
        x = self.xx[-1]#self.xx[-2]
        u = self.uu[-1]

        system_matrices = self.sys_lin()

        Cxx, Cuu, Cxu, cx, cu = self.cost_lin(x, 0*u)

        Cux = Cxu.T

        Ct = np.block([[Cxx, Cxu], [Cux, Cuu]])

        CuuInv = np.linalg.inv(Cuu + self.mu * np.eye(self.uDim))

        Kt = -np.matmul(CuuInv, Cux)
        kt = -np.matmul(CuuInv, cu)

        # DARE in Vt, vt
        Vt = self.finalcost*Cxx #+ np.matmul(Cxu, Kt) + np.matmul(Kt.T, Cux) + np.matmul(Kt.T, np.matmul(Cuu, Kt))
        vt = self.finalcost*cx #+ np.matmul(Cxu, kt) + np.matmul(Kt.T, cu) + np.matmul(Kt.T, np.matmul(Cuu, kt))
        dV1 = np.matmul(kt.T, np.matmul(Cuu, kt))
        dV2 = np.matmul(kt.T, cu)

        self.KK = []#[Kt]
        self.kk = []#[kt]

        V1 = [dV1]
        V2 = [dV2]

        success = True

        for i in range(self.steps-1, -1, -1):#range(self.steps - 1):
            x = self.xx[i]#[-i - 3]
            u = self.uu[i]#[-i - 2]

            Cxx, Cuu, Cxu, cx, cu = self.cost_lin(x, u)

            ct = np.block([[cx], [cu]])

            Ct = np.block([[Cxx, Cxu], [Cux, Cuu]])

            # Linerisierung
            Ft, ft = system_matrices[i]#[-i - 2]

            Qt = Ct + np.matmul(Ft.T, np.matmul(Vt + self.mu * np.eye(self.xDim), Ft))

            Qxx, Qxu, Qux, Quu = unpackBlockMatrix(Qt, self.xDim, self.uDim)

            qt = ct + np.matmul(Ft.T, np.matmul(Vt, ft)) + np.matmul(Ft.T, vt)
            qx = qt[:self.xDim]
            qu = qt[self.xDim:]

            '''
            if np.any(np.linalg.eigvals(Quu) <= 0):
                success = False
            '''
            try:
                np.linalg.cholesky(Quu)

            except np.linalg.LinAlgError as e:
                success = False
                break
            QuuInv = np.linalg.inv(Quu + self.mu * np.eye(self.uDim))

            if QuuInv[0, 0] < 0:
                print(QuuInv[0, 0])

            Kt = -np.matmul(QuuInv, Qux)
            kt = -np.matmul(QuuInv, qu)

            Vt = Qxx + np.matmul(Kt.T, np.matmul(Quu, Kt)) + np.matmul(Kt.T, Qux) + np.matmul(Qux.T, Kt)
            vt = qx + np.matmul(Kt.T, np.matmul(Quu, kt)) + np.matmul(Kt.T, qu) + np.matmul(Qux.T, kt)

            dV2 = 0.5 * np.matmul(kt.T, np.matmul(Quu, kt))
            dV1 = np.matmul(kt.T, qu)
            # save Kt, kt
            self.KK.insert(0, Kt)
            self.kk.insert(0, kt)

            V1.insert(0, dV1)
            V2.insert(0, dV2)

        return V1, V2, success

    def run_controller(self):
        # Todo: implement saving and loading the control gains, implement running them
        pass


    def run_optim(self):
        self.mu = 1.0
        success_gradient = False
        # later run_episode
        for _ in range(self.max_iters):
            try:
                success_bw = False
                while not success_bw:
                    V1, V2, success_bw = self.backward_pass()
                    if not success_bw:
                        # print('Backward successfull')
                        print('diverged')
                        # increase mu
                        self.increase_mu()
                        break
                # check for gradient
                g_norm = np.mean(np.max(np.abs(np.array(self.kk).reshape((len(self.kk), self.uDim))/(np.abs(self.uu) + 1)), axis=0))
                if g_norm < self.tolGrad and self.mu < 1e-5:
                    success_gradient = True

                success_fw = False
                if success_bw:
                    # Line-search
                    for alpha in self.alphas:
                        xx, uu, cost = self.forward_pass(alpha)
                        dcost = self.cost - cost
                        expected = -alpha * (np.sum(V1) + alpha * np.sum(V2))
                        if expected > 0:
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
                    if not success_fw:
                        # increase mu
                        self.increase_mu()
                        print('Forward not successfull')
                print('Iteration ', _, '/ Cost: ', cost)
            except KeyboardInterrupt:
                self.plot()
                plt.show()
                self.animation()

            if success_fw:
                # print('Iteration successfull')
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
        print('Iterations ', _, '/ Final Cost-to-Go: ', self.cost)

        return self.xx, self.uu, self.cost


    def init_trajectory(self):
        # random trajectory
        for _ in range(self.steps):
            u = np.random.uniform(0.01, 0.01, self.uDim)
            u = self.agent.control(self.dt, u)
            # necessary to store control in agents history
            if self.fastForward:
                c = self.environment.fast_step(self.dt, u)
            else:
                c = self.environment.step(self.dt, u)
            self.cost += c*self.dt
        self.xx = self.environment.history
        self.uu = self.agent.history[1:]
        pass

    def sys_lin(self):
        # 1st order taylor expansion of the system dynamics along a trajectory
        return [self.linearization(x, u) for x, u in zip(self.xx, self.uu)]

    def linearization(self, x, u):

        A, B = system_linearization(lambda xx, uu: self.environment.ode(None, xx, uu), x, u)

        Ad, Bd = c2d(A, B, self.dt)
        #Ad = A*self.dt + np.eye(self.xDim)
        #Bd = B*self.dt
        #Ft = np.block([[self.Ad(x, u), self.Bd(x, u)]])
        Ft = np.block([[Ad, Bd]])
        ft = np.zeros((self.xDim,1))#self.dt *0* np.expand_dims(self.environment.ode(None, x, u), 0).T
        return Ft, ft

    def cost_init(self):
        # 2nd order taylor expansion of the cost function along a trajectory
        #self.environment.

        xx = sp.symbols('x1:'+str(self.xDim+1))
        uu = sp.symbols('u1:' + str(self.uDim + 1))

        c = self.cost_fnc(xx, uu)
        cc = sp.Matrix([[c]])
        cx = cc.jacobian(xx)
        cu = cc.jacobian(uu)
        Cxx = cx.jacobian(xx)
        Cuu = cu.jacobian(uu)
        Cxu = cx.jacobian(uu)

        self.cx = sp.lambdify((xx, uu), cx.T)
        self.cu = sp.lambdify((xx, uu), cu.T)
        self.Cxx = sp.lambdify((xx, uu), Cxx)
        self.Cuu = sp.lambdify((xx, uu), Cuu)
        self.Cxu = sp.lambdify((xx, uu), Cxu)

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
        return self.Cxx(x, u), self.Cuu(x, u), self.Cxu(x, u), self.cx(x, u), self.cu(x, u)

    def decrease_mu(self):
        self.mu_d = min(1 / self.mu_d0, self.mu_d / self.mu_d0)
        if self.mu * self.mu_d > self.mu_max:
            self.mu *= self.mu_d
        elif self.mu * self.mu_d < self.mu_min:
            self.mu = 0.0
        pass

    def increase_mu(self):
        self.mu_d = max(self.mu_d0, self.mu_d0 * self.mu_d)
        self.mu = max(self.mu_min, self.mu * self.mu_d)
        pass

    def plot(self):
        self.environment.plot()
        plt.savefig('results/iLQR_environment')
        self.agent.plot()
        plt.savefig('results/iLQR_agent')
        plt.close('all')

    def animation(self):
        ani = self.environment.animation(0, self.cost)
        if ani != None:
            ani.save('results/iLQR_animation.mp4', fps=1/self.dt)
        plt.close('all')