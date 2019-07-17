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
from pygent.algorithms.nmpc import NMPC
from pygent.nn_models import NNDynamics

class MBRL(Algorithm):

    def __init__(self, environment, t, dt,
                 plotInterval=5,
                 nData=1e6,
                 path='../results/mbrl/',
                 checkInterval=50,
                 evalPolicyInterval=100,
                 warm_up=10000,
                 dyn_lr=1e-3,
                 batch_size=512,
                 training_epochs=60,
                 data_ratio = 9,
                 aggregation_interval=10,
                 fcost=None, horizon=None, use_mpc_plan=True):
        xDim = environment.xDim
        oDim = environment.oDim
        uDim = environment.uDim
        uMax = environment.uMax
        if horizon == None:
            if use_mpc_plan == True:
                horizon = t
            else:
                horizon = 100*dt
        self.nn_dynamics = NNDynamics(xDim, uDim, oDim=oDim, xIsAngle=self.environment.xIsAngle) # neural network dynamics
        self.optim = torch.optim.Adam(self.nn_dynamics.parameters(), lr=dyn_lr)
        nn_environment = StateSpaceModel(self.ode, environment.cost, environment.x0, uDim, dt)
        nn_environment.uMax = uMax
        #agent = MPCAgent(uDim, nn_environment, horizon, dt, path)
        self.nmpc_algorithm = NMPC(copy.deepcopy(nn_environment), copy.deepcopy(nn_environment), t, dt, horizon,
                                   path=path, fcost=fcost, fastForward=True, init_optim=False)
        super(MBRL, self).__init__(environment, self.nmpc_algorithm.agent, t, dt)
        self.D_rand = DataSet(nData)
        self.D_RL = DataSet(nData)
        self.plotInterval = plotInterval  # inter
        self.evalPolicyInterval = evalPolicyInterval
        self.checkInterval = checkInterval  # checkpoint interval
        self.path = path
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.data_ratio = data_ratio
        self.aggregation_interval = aggregation_interval
        self.use_mpc_plan=use_mpc_plan
        self.dynamics_first_trained = False

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
        rhs = self.nn_dynamics.ode(x, u)
        return rhs

    def run_episode(self):
        """ Run a training episode. If terminal state is reached, episode stops."""

        print('Started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        cost = []  # list of incremental costs
        disc_cost = [] # discounted cost

        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()
        self.agent.traj_optimizer.environment.reset(self.environment.x)
        self.agent.init_optim()

        for i, t in enumerate(tt):
            # agent computes control/action

            noise = np.random.normal(loc=0.0, scale=0.005, size=self.environment.uDim)
            if self.use_mpc_plan:
                u = self.agent.take_action_plan(self.dt, self.environment.x, i) + noise
            else:
                u = self.agent.take_action(self.dt, self.environment.x) + noise
            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)
            disc_cost.append(c)

            # store transition in data set (x_, u, x, c)
            transition = ({'x_': self.environment.x_, 'u': self.agent.u, 'x': self.environment.x,
                           'o_': self.environment.o_, 'o': self.environment.o,
                           'c': [c], 't': [self.environment.terminated]})

            # add sample to data set
            self.D_RL.force_add_sample(transition)

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

    def random_episode(self):
        print('Warmup. Started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        cost = []  # list of incremental costs
        disc_cost = []  # discounted cost

        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()

        for i, t in enumerate(tt):
            # agent computes control/action
            u = self.agent.take_random_action(self.dt)
            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)
            disc_cost.append(c)

            # store transition in data set (x_, u, x, c)
            transition = ({'x_': self.environment.x_, 'u': self.agent.u, 'x': self.environment.x,
                           'o_': self.environment.o_, 'o': self.environment.o,
                           'c': [c], 't': [self.environment.terminated]})

            # add sample to data set
            self.D_rand.force_add_sample(transition)
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
            u = self.agent.take_action(self.dt, self.environment.x) + self.uMax*np.random.normal(0, 0.005, self.uDim)

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
        for steps in range(1, int(n) + 1):
            if self.D_rand.data.__len__()<self.warm_up:#self.batch_size:
                self.random_episode()
            else:
                if not self.dynamics_first_trained:
                    self.train_dynamics()
                    self.dynamics_first_trained = True
                if steps % self.aggregation_interval == 0:
                    self.train_dynamics()
                self.run_episode()
            # plot environment after episode finished
            print('Samples: ', self.D_rand.data.__len__(), self.D_RL.data.__len__())
            if self.episode % 10 == 0:
                self.learning_curve()
            if self.episode % self.checkInterval == 0:
                self.save()
                # if self.meanCost[-1] < 0.01: # goal reached
        pass

    def save(self):
        """ Save neural network parameters and data set. """

        # save network parameters
        torch.save({'nn_dynamics': self.nn_dynamics.state_dict()}, self.path + 'data/checkpoint.pth')

        # save data set
        self.D_rand.save(self.path + 'data/dataSet_D_rand.p')
        self.D_RL.save(self.path + 'data/dataSet_D_RL.p')
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
        if os.path.isfile(self.path + 'data/dataSet_D_rand.p'):
            self.D_rand.load(self.path + 'data/dataSet_D_rand.p')
            print('Loaded data set D_rand!')
        else:
            print('No data set found!')

        # load data set
        if os.path.isfile(self.path + 'data/dataSet_D_RL.p'):
            self.D_RL.load(self.path + 'data/dataSet_D_RL.p')
            print('Loaded data set D_rand!')
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
        self.agent.plot()
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_agent.pdf')
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
        x = np.linspace(1, self.D_rand.data.__len__()+self.D_RL.data.__len__(), self.episode-1)
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
        # todo: save learning curve data
        plt.close('all')
        pass

    def train_dynamics(self):
        training_data_set = copy.deepcopy(self.D_rand)
        for _ in range(self.data_ratio):
            training_data_set.data += self.D_RL.data
        self.training(training_data_set)
        torch.save(self.nn_dynamics.state_dict(), self.path + '.pth')

    def training(self, dataSet):
        # loss function (mean squared error)
        criterion = nn.MSELoss()
        self.nn_dynamics.eval()
        self.set_moments(dataSet)
        # create training data/targets
        for epoch in range(self.training_epochs):
            running_loss = 0.0
            for iter, batch in enumerate(dataSet.shuffled_batches(self.batch_size)):
                x_Inputs, xInputs, fOutputs = self.training_data(batch)
                # definition of loss functions
                loss = criterion(fOutputs, (xInputs - x_Inputs) / self.dt)
                # train
                self.optim.zero_grad()  # delete gradients
                loss.backward()  # error back-propagation
                self.optim.step()  # gradient descent step
                running_loss += loss.item()
                # self.eval() # eval mode on (batch normalization)
        print(epoch, running_loss / max(1, iter))
        pass

    def training_data(self, batch):
        x_Inputs = torch.Tensor([sample['x_'] for sample in batch])
        xInputs = torch.Tensor([sample['x'] for sample in batch])
        o_Inputs = torch.Tensor([sample['o_'] for sample in batch])
        uInputs = torch.Tensor([sample['u'] for sample in batch])
        fOutputs = self.nn_dynamics(o_Inputs, uInputs)
        return x_Inputs, xInputs, fOutputs



