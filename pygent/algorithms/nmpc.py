import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
import time
import os
import inspect
from shutil import copyfile
import scipy as sci

# pygent
from pygent.agents import Agent
from pygent.algorithms.core import Algorithm
from pygent.algorithms.ilqr import iLQR
from pygent.helpers import OUnoise

class NMPC(Algorithm):
    def __init__(self, environment, mpc_environment, t, dt, horizon,
                 maxIters=500,
                 step_iterations=1,
                 tolGrad=1e-4,
                 tolFun=1e-7,
                 fastForward=False,
                 path='../results/nmpc/',
                 fcost=None,
                 constrained=True,
                 save_interval=50,
                 init_optim=True,
                 finite_diff=False,
                 ilqr_print=False,
                 ilqr_save=False,
                 noise_gain=0.005,
                 add_noise=False,
                 ou_theta=0.15,
                 ou_sigma=0.2):
        """

        Args:
            environment (Environment):
            mpc_environment (Environment):
            t (float): simulation time
            horizon (float):
            dt (float): step-size
            maxIters (int): maximum number of iterations
            tolGrad (float):
            tolFun (float):
            fastForward (bool): if True, use Euler forward integration
            path (string): directory for saving time-varying controllers and trajectories
            fcost (function): Final cost function. c = fcost(x_N)
            constrained (bool): if True, control input is constrained
        """
        self.path = path
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path + 'plots/'):
            os.makedirs(path + 'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')
        copyfile(inspect.stack()[-1][1], path + 'exec_script.py')
        self.mpc_environment = mpc_environment
        agent = MPCAgent(mpc_environment, horizon, dt, path,
                         init_iterations=maxIters,
                         step_iterations=step_iterations,
                         tolGrad = tolGrad,
                         tolFun=tolFun,
                         fcost=fcost,
                         constrained=constrained,
                         fastForward=fastForward,
                         printing=ilqr_print,
                         saving=ilqr_save,
                         init_optim=init_optim,
                         finite_diff=finite_diff,
                         noise_gain=noise_gain,
                         add_noise=add_noise,
                         save_interval=save_interval,
                         ou_sigma=ou_sigma,
                         ou_theta=ou_theta)

        super(NMPC, self).__init__(environment, agent, t, dt)
        self.save_interval = save_interval


    def run(self):
        start_time = time.time()
        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()
        cost = []
        tt = np.arange(0, self.t, self.dt)

        for i, t in enumerate(tt):
            print('Step ', i+1, '/', len(tt))
            # agent computes control/action
            u = self.agent.take_action(self.dt, self.environment.x)

            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)
            if i % self.save_interval == 0:
                self.plot()
        print('Cost: %.2f | Runtime: %.2f min.' % (np.sum(cost), (time.time() - start_time) / 60))
        self.plot()
        self.animation()

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


class MPCAgent(Agent):
    def __init__(self, environment, horizon, dt, path,
                 init_iterations=500,
                 step_iterations=1,
                 fcost=None,
                 constrained=True,
                 fastForward=False,
                 tolGrad = 1e-4,
                 tolFun = 1e-7,
                 save_interval=10,
                 printing=True,
                 saving=True,
                 init_optim = True,
                 finite_diff=True,
                 noise_gain=0.05,
                 add_noise=False,
                 ou_theta=0.15,
                 ou_sigma=0.2):
        super(MPCAgent, self).__init__(environment.uDim)
        self.traj_optimizer = iLQR(environment, horizon, dt,
                                   path=path,
                                   fcost=fcost,
                                   constrained=constrained,
                                   fastForward=fastForward,
                                   save_interval=save_interval,
                                   tolGrad = tolGrad,
                                   tolFun = tolFun,
                                   printing=printing,
                                   finite_diff=finite_diff,
                                   file_prefix='ilqr_',
                                   init=False,
                                   saving=saving,
                                   reset_mu=True)
        self.uMax = self.traj_optimizer.environment.uMax
        self.init_iterations = init_iterations
        self.step_iterations = step_iterations
        self.x0 = environment.x
        if init_optim==True:
            self.init_optim(environment.x)
        self.noise_gain = noise_gain
        self.add_noise = add_noise
        self.action_noise = OUnoise(self.uDim, dt, theta=ou_theta, sigma=ou_sigma)



        #self.traj_optimizer.plot()

    def init_optim(self, x0, init=True):
        self.traj_optimizer.reset_mu = True
        self.traj_optimizer.environment.reset(x0)
        self.traj_optimizer.environment.x0 = x0
        self.traj_optimizer.init = init
        self.traj_optimizer.reset()
        self.traj_optimizer.max_iters = self.init_iterations
        print('Running inital optimization.')
        self.traj_optimizer.run_optim()
        self.traj_optimizer.max_iters = self.step_iterations
        self.traj_optimizer.reset_mu = True
        pass


    def take_action(self, dt, x):
        """ Compute the control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): control/action
        """
        self.traj_optimizer.environment.x0 = x
        self.traj_optimizer.run_optim()
        kk = self.traj_optimizer.kk[0].T[0]
        KK = self.traj_optimizer.KK[0]
        uu = self.traj_optimizer.uu[0]
        xx = self.traj_optimizer.xx[0]
        alpha = self.traj_optimizer.current_alpha

        u = KK @ (x - xx) + uu + alpha * kk + self.add_noise*self.noise()
        self.u = np.clip(u, -self.uMax, self.uMax)

        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        self.shift_planner()
        return self.u


    def take_action_plan(self, dt, i):
        """ Compute the control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): control/action
        """
        kk = self.traj_optimizer.kk[i].T[0]
        alpha = self.traj_optimizer.current_alpha
        u = self.traj_optimizer.uu[i]  + alpha*kk + self.add_noise*self.noise()
        self.u = np.clip(u, -self.uMax, self.uMax)
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def take_action_plan_feedback(self, dt, x, i):
        """ Compute the control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): control/action
        """
        if i < self.traj_optimizer.KK.__len__()-1:
            kk = self.traj_optimizer.kk[i].T[0]
            KK = self.traj_optimizer.KK[i]
            uu = self.traj_optimizer.uu[i]
            xx = self.traj_optimizer.xx[i]
            alpha = self.traj_optimizer.current_alpha
            u = KK@(x - xx)+ uu + alpha*kk + self.add_noise*self.noise()
        else:
            KK = self.traj_optimizer.KK[-1]
            uu = self.traj_optimizer.uu[-1]
            # determine x_ref from final cost
            f = lambda x: self.traj_optimizer.environment.cost(x, 0*uu, None, None, np)
            equil = sci.optimize.minimize(f, x)
            xx = equil.x
            u = KK@(x - xx) + uu 
        self.u = np.clip(u, -self.uMax, self.uMax)

        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u


    def shift_planner(self):
        dt = self.traj_optimizer.environment.dt
        u0 = self.traj_optimizer.uu[0]
        x0 = self.traj_optimizer.xx[0]
        x1 = self.traj_optimizer.xx[1]
        xN = self.traj_optimizer.xx[-1]
        c0 = self.traj_optimizer.environment.cost(x0, u0, x1, None, np)*dt
        cN = self.traj_optimizer.fcost_fnc(xN, np)*dt
        self.traj_optimizer.cost -= c0 + cN

        self.traj_optimizer.uu = np.roll(self.traj_optimizer.uu, -1, axis=0)
        self.traj_optimizer.uu[-1] = self.traj_optimizer.uu[-2]*0
        self.traj_optimizer.xx = np.roll(self.traj_optimizer.xx, -1, axis=0)
        self.traj_optimizer.kk[-1] = self.traj_optimizer.kk[-1]*0
        self.traj_optimizer.KK[-1] = self.traj_optimizer.KK[-1]*0
        u = self.traj_optimizer.uu[-1]
        c = self.traj_optimizer.environment.step(u)
        cN = self.traj_optimizer.fcost_fnc(self.traj_optimizer.environment.x, np)*dt
        self.traj_optimizer.cost += c + cN
        self.traj_optimizer.xx[-1] = self.traj_optimizer.environment.x
        pass

    def take_random_action(self, dt, ou_noise=True):
        """ Compute a random control/action (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): noisy control/action
        """
        if ou_noise:
            u = self.action_noise.sample()*self.uMax
            self.u = np.clip(u, -self.uMax, self.uMax)
        else:
            self.u = np.random.uniform(-self.uMax, self.uMax, self.uDim)
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def noise(self):
        return self.uMax * np.random.normal(0, self.noise_gain, self.uDim)
