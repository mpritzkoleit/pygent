import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
import time
import os
import inspect
from shutil import copyfile

# pygent
from pygent.agents import Agent
from pygent.algorithms.core import Algorithm
from pygent.algorithms.ilqr import iLQR

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
                 save_interval=10,
                 init_optim=True,
                 finite_diff=False,
                 ilqr_print=False,
                 noise_gain=0.01,
                 add_noise=False):
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
                         init_optim=init_optim,
                         finite_diff=finite_diff,
                         noise_gain=noise_gain,
                         add_noise=add_noise)

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
                 init_optim = True,
                 finite_diff=True,
                 noise_gain=0.05,
                 add_noise=False):
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
                                   file_prefix='ilqr_')
        self.uMax = self.traj_optimizer.environment.uMax
        self.init_iterations = init_iterations
        self.step_iterations = step_iterations
        self.x0 = environment.x0
        if init_optim==True:
            self.init_optim()
        self.noise_gain = noise_gain
        self.add_noise = add_noise

        #self.traj_optimizer.plot()

    def init_optim(self):
        self.traj_optimizer.environment.reset(self.x0)
        self.traj_optimizer.mu = 1.
        self.traj_optimizer.mu_d = 1.
        if self.traj_optimizer.KK != []:
            self.traj_optimizer.cost *= 200 # double cost, because forward pass will lead to the same cost
            self.traj_optimizer.forward_pass(self.traj_optimizer.current_alpha)
        else:
            self.traj_optimizer.init_trajectory()
        self.traj_optimizer.max_iters = self.init_iterations
        print('Running inital optimization.')
        self.traj_optimizer.run_optim()
        self.traj_optimizer.max_iters = self.step_iterations
        pass

    def clear(self):
        self.traj_optimizer.environment.reset(self.x0)
        self.traj_optimizer.KK = []
        self.traj_optimizer.kk = []
        self.traj_optimizer.mu = 1.
        self.traj_optimizer.mu_d = 1.

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


    def take_action_plan(self, dt, x, i):
        """ Compute the control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): control/action
        """

        #kk = self.traj_optimizer.kk[i].T[0]
        #KK = self.traj_optimizer.KK[i]
        u = self.traj_optimizer.uu[i] + self.add_noise*self.noise()
        self.u = np.clip(u, -self.uMax, self.uMax)
        #xx = self.traj_optimizer.xx[i]
        #alpha = self.traj_optimizer.current_alpha
        #self.u = KK@(x - xx)+ uu + alpha*kk
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

        kk = self.traj_optimizer.kk[i].T[0]
        KK = self.traj_optimizer.KK[i]
        uu = self.traj_optimizer.uu[i]
        xx = self.traj_optimizer.xx[i]
        alpha = self.traj_optimizer.current_alpha

        u = KK@(x - xx)+ uu + alpha*kk + self.add_noise*self.noise()
        self.u = np.clip(u, -self.uMax, self.uMax)

        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u


    def shift_planner(self):
        self.traj_optimizer.uu = np.roll(self.traj_optimizer.uu, -1, axis=0)
        self.traj_optimizer.uu[-1] = self.traj_optimizer.uu[-2]
        self.traj_optimizer.xx = np.roll(self.traj_optimizer.xx, -1, axis=0)
        self.traj_optimizer.kk[-1] = self.traj_optimizer.kk[-1]*0
        self.traj_optimizer.KK[-1] = self.traj_optimizer.KK[-1]*0
        u = self.traj_optimizer.uu[-1]
        self.traj_optimizer.environment.step(u)
        self.traj_optimizer.xx[-1] = self.traj_optimizer.environment.x
        self.traj_optimizer.environment.reset(self.traj_optimizer.xx[0])
        pass

    def take_random_action(self, dt):
        """ Compute a random control/action (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): noisy control/action
        """

        self.u = np.random.uniform(-self.uMax, self.uMax, self.uDim)
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def noise(self):
        return self.uMax * np.random.normal(0, self.noise_gain, self.uDim)
