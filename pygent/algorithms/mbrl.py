import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pickle
import inspect
from shutil import copyfile
import copy

# pygent
from pygent.agents import Agent
from pygent.data import DataSet
from pygent.environments import StateSpaceModel
from pygent.algorithms.core import Algorithm
from pygent.algorithms.ilqr import iLQR
from pygent.nn_models import NNDynamics

class MBRL(Algorithm):

    def __init__(self, environment, t, dt, plotInterval=10, nData=1e6, path='../results/mbrl/', checkInterval=50,
                 evalPolicyInterval=100, warm_up=10000, dyn_lr=1e-3, batch_size=512, aggregation_interval=10,
                 fcost=None, horizon=None):
        xDim = environment.xDim
        uDim = environment.uDim
        uMax = environment.uMax
        if horizon == None:
            horizon = 10*dt
        self.nn_dynamics = NNDynamics(xDim, uDim) # neural network dynamics
        self.optim = torch.optim.Adam(self.nn_dynamics.parameters(), lr=dyn_lr)
        nn_environment = StateSpaceModel(self.ode, environment.cost, environment.x0, uDim, dt)
        nn_environment.uMax = uMax
        #agent = MPCAgent(uDim, nn_environment, horizon, dt, path)
        agent = MPCAgent(uDim, copy.deepcopy(environment), horizon, dt, path)
        super(MBRL, self).__init__(environment, agent, t, dt)
        self.R = DataSet(nData)
        self.R_RL = DataSet(nData)
        self.plotInterval = plotInterval  # inter
        self.evalPolicyInterval = evalPolicyInterval
        self.checkInterval = checkInterval  # checkpoint interval
        self.path = path
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.dyn_steps_train = 60
        self.aggregation_interval = aggregation_interval

        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path + 'plots/'):
            os.makedirs(path + 'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')
        copyfile(inspect.stack()[-1][1], path + 'exec_script.py')
        self.expCost = []
        self.episode_steps = []

    def ode(self, t, x, u):
        if type(u)==type(list()):
            u = np.array([u])
        if type(x)==type(list()):
            x = np.array([x])
        if u.ndim == 1:
            u = u.reshape(1, len(u))
        if x.ndim == 1:
            x = x.reshape(1, len(x))
        rhs = self.nn_dynamics(torch.Tensor(x), torch.Tensor(u)).detach().numpy()
        return rhs[0]

    def run_episode(self):
        """ Run a training episode. If terminal state is reached, episode stops."""

        print('Started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        cost = []  # list of incremental costs
        disc_cost = [] # discounted cost

        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()
        if self.R_RL.data.__len__() >= self.batch_size:
            self.agent.traj_optimizer.environment.reset(self.environment.x)
            self.agent.init_optim()

        for i, t in enumerate(tt):
            # agent computes control/action
            if self.R_RL.data.__len__() >= self.batch_size:
                u = self.agent.take_action(self.dt, self.environment.x, i)
            else:
                u = self.agent.take_random_action(self.dt)
            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)
            disc_cost.append(c)

            # store transition in data set (x_, u, x, c)
            transition = ({'x_': self.environment.x_, 'u': self.agent.u, 'x': self.environment.x,
                           'c': [c], 't': [self.environment.terminated]})

            # add sample to data set
            if self.R.data.__len__() < self.warm_up:
                self.R.force_add_sample(transition)
            else:
                self.R_RL.force_add_sample(transition)

            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        self.totalCost.append(np.sum(disc_cost))
        self.episode_steps.append(i)
        self.episode += 1
        pass

    def run_controller(self, x0):
        """ Run an episode, where the policy network is evaluated. """

        print('Started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        cost = []  # list of incremental costs

        # reset environment/agent to initial state, delete history
        self.environment.reset(x0)
        self.agent.reset()

        for i, t in enumerate(tt):
            # agent computes control/action
            u = self.agent.take_action(self.dt, self.environment.x) + self.uMax*np.random.normal(-self.uMax, 0.005, self.uDim)

            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)

            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break
        pass

    def run_learning(self, n):
        """ Learning process.

            Args:
                n (int): number of episodes
        """

        for k in range(1, n + 1):
            if self.R.data.__len__()>self.batch_size:
                for i in range(1):
                    print('train dynamics')
                    self.train_dynamics(self.R)
                    if self.R_RL.data.__len__() > self.batch_size:
                        for i in range(9):
                            self.train_dynamics(self.R_RL)
            for i in range(2):
                self.run_episode()               
            # plot environment after episode finished
            print('Samples: ', self.R.data.__len__(), self.R_RL.data.__len__())
            if k % 10 == 0:
                self.learning_curve()
            if k % self.checkInterval == 0:
                self.save()
                # if self.meanCost[-1] < 0.01: # goal reached
            if k % self.plotInterval == 0:
                self.plot()
                self.animation()
        pass

    def save(self):
        """ Save neural network parameters and data set. """

        # save network parameters
        torch.save({'nn_dynamics': self.nn_dynamics.state_dict()}, self.path + 'data/checkpoint.pth')

        # save data set
        self.R.save(self.path + 'data/dataSet.p')

        # save learning curve data
        learning_curve_dict = {'totalCost': self.totalCost, 'meanCost':self.meanCost,
                               'expCost': self.expCost, 'episode_steps': self.episode_steps}

        pickle.dump(learning_curve_dict, open(self.path + 'data/learning_curve.p', 'wb'))
        print('Network parameters, data set and learning curve saved.')
        pass

    def load(self):
        """ Load neural network parameters and data set. """

        # load network parameters
        if os.path.isfile(self.path + 'data/checkpoint.pth'):
            checkpoint = torch.load(self.path + 'data/checkpoint.pth')
            self.nn_dynamics.load_state_dict(checkpoint['nn_dynamics'])
            print('Loaded neural network parameters!')
        else:
            print('Could not load neural network parameters!')

        # load data set
        if os.path.isfile(self.path + 'data/dataSet.p'):
            self.R.load(self.path + 'data/dataSet.p')
            print('Loaded data set!')
        else:
            print('No data set found!')

        # load learning curve
        if os.path.isfile(self.path + 'data/learning_curve.p'):
            learning_curve_dict = pickle.load(open(self.path + 'data/learning_curve.p', 'rb'))
            self.meanCost = learning_curve_dict['meanCost']
            self.totalCost = learning_curve_dict['totalCost']
            self.expCost = learning_curve_dict['expCost']
            self.episode_steps = learning_curve_dict['episode_steps']
            self.episode = self.meanCost.__len__() + 1
            print('Loaded learning curve data!')
        else:
            print('No learning curve data found!')
        self.run_controller(self.environment.x0)
        pass

    def plot(self):
        """ Plots the environment's and agent's history. """

        self.environment.plot()
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pdf')
        try:
            plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pgf')
        except:
            pass
        self.agent.plot()
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_agent.pdf')
        try:
            plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_agent.pgf')
        except:
            pass
        plt.close('all')
        pass

    def animation(self):
        """ Animation of the environment (if available). """

        ani = self.environment.animation()
        if ani != None:
            try:
                ani.save(self.path + 'animations/' + str(self.episode - 1) + '_animation.mp4', fps=1 / self.dt)
            except:
                ani.save(self.path + 'animations/' + str(self.episode - 1) + '_animation.gif', fps=1 / self.dt)
        plt.close('all')
        pass

    def learning_curve(self):
        """ Plot of the learning curve. """

        fig, ax = plt.subplots(2, 1, dpi=150, sharex=True, figsize=(5.56, 3.44))

        #x = np.arange(1, self.episode)
        x = np.linspace(1, self.R.data.__len__(), self.episode-1)
        x = np.cumsum(self.episode_steps)

        ax[0].step(x, self.meanCost, 'b', lw=1, label=r'$\frac{1}{N}\sum_{k=0}^N c_k$')
        ax[0].legend(loc='center', bbox_to_anchor=(1.15, .5), ncol=1, shadow=True)
        ax[0].grid(True)
        ax[0].ticklabel_format(axis='both', style='sci', scilimits=(-3,4), useMathText=True)
        ax[1].step(x, self.totalCost, 'b', lw=1, label=r'$\sum_{k=0}^N\gamma^k c_k$')
        ax[1].grid(True)
        ax[1].legend(loc='center', bbox_to_anchor=(1.15, .5), ncol=1, shadow=True)
        ax[1].ticklabel_format(axis='both', style='sci',scilimits=(-3,5), useMathText=True)
        plt.rc('font', family='serif')
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.savefig(self.path + 'learning_curve.pdf')
        try:
            plt.savefig(self.path + 'learning_curve.pgf')
        except:
            pass
        # todo: save learning curve data
        plt.close('all')
        pass

    def train_dynamics(self, dataSet):
        # loss function (mean squared error)
        criterion = nn.MSELoss()
        # create training data/targets
        batch = dataSet.minibatch(self.batch_size)

        for epoch in range(self.dyn_steps_train):

            x_Inputs, xInputs, fOutputs = self.training_data(batch)
            # definition of loss functions
            loss = criterion(fOutputs, (xInputs - x_Inputs)/self.dt)
            running_loss = 0.0
            # train
            self.optim.zero_grad()  # delete gradients
            loss.backward()  # error back-propagation
            self.optim.step()  # gradient descent step
            running_loss += loss.item()

            #self.eval() # eval mode on (batch normalization)
        pass

    def training_data(self, batch):
        x_Inputs = torch.Tensor([sample['x_'] for sample in batch])
        xInputs = torch.Tensor([sample['x'] for sample in batch])
        uInputs = torch.Tensor([sample['u'] for sample in batch])
        fOutputs = self.nn_dynamics(x_Inputs, uInputs)
        return x_Inputs, xInputs, fOutputs



class MPCAgent(Agent):
    def __init__(self, uDim, environment, horizon, dt, path, init_iterations=40, fcost=None, constrained=True, fastForward=False):
        super(MPCAgent, self).__init__(uDim)
        self.traj_optimizer = iLQR(environment, horizon, dt, path=path, fcost=fcost, constrained=constrained,fastForward=fastForward)
        self.uMax = self.traj_optimizer.environment.uMax
        self.init_iterations = init_iterations
        #self.init_optim()
        #self.traj_optimizer.plot()

    def init_optim(self):
        self.traj_optimizer.max_iters = self.init_iterations
        #self.traj_optimizer.init_trajectory()
        self.traj_optimizer.run_optim()
        self.traj_optimizer.max_iters = 1
        pass

    def take_action(self, dt, x, idx):
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
        uu = self.traj_optimizer.uu[0]
        alpha = self.traj_optimizer.current_alpha
        self.u = uu + alpha*kk
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        self.shift_planner()
        return self.u


    def take_action_plan(self, dt, x, idx):
        """ Compute the control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): control/action
        """

        #self.traj_optimizer.environment.x0 = x
        #self.traj_optimizer.run_optim()
        kk = self.traj_optimizer.kk[idx].T[0]
        uu = self.traj_optimizer.uu[idx]
        alpha = self.traj_optimizer.current_alpha
        self.u = uu + alpha*kk
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def shift_planner(self):
        self.traj_optimizer.uu[0:-1] = self.traj_optimizer.uu[1:]
        self.traj_optimizer.xx[0:-1] = self.traj_optimizer.xx[1:]
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
