import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
import random
random.seed(0)
import os
import inspect
from shutil import copyfile


# pygent
from pygent.agents import Agent
from pygent.data import DataSet
from pygent.nn_models import MLP
from pygent.algorithms.core import Algorithm
from pygent.environments import observation

class QNetwork(Agent):
    """ Q-Network (Multi-Layer-Perceptron)
        Q(x,u) -> R

        mu(x) = argmin_u*(Q(x[k],u*)


    Attributes:
        controls (array): control/actions that the agent can choose from
        netStructure (array): array that describes layer structure (i.e. [1, 10, 10, 1])
        eps (float [0, 1]): with probability eps a random action/control is returned
    """

    def __init__(self, controls, netStructure, eps, gamma, xGoal):
        super(QNetwork, self).__init__(1)
        self.controls = controls
        self.qNetwork = MLP(netStructure)
        self.eps = eps
        self.gamma = gamma
        self.optimizer = torch.optim.Rprop(self.qNetwork.parameters())
        self.xGoal = xGoal # goal state
        # implement neural network in pytorch


    def train(self, dataSet):
        # loss function (mean squared error)
        criterion = nn.MSELoss()

        # training data
        inputs, targets = self.training_data(dataSet)

        # artificial hint-to-goal training data (fixes the Q-Netork ouptut to 0 in the goal region)
        inputs_htg, targets_htg = self.artificial_data(max(int(len(dataSet.data)/10), 1))

        # add hint-to-goal data to training data
        # bn = nn.BatchNorm1d(self.qNetwork.netStructure[0])
        inputs = torch.cat((inputs, inputs_htg), 0)
        targets = torch.cat((targets, targets_htg), 0)

        for epoch in range(300):  # loop over the dataset multiple times
            running_loss = 0.0

            outputs = self.qNetwork(inputs)

            # backward + optimize
            loss = criterion(outputs, targets)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            loss.backward(retain_graph=True)
            self.optimizer.step()
        pass

    def take_action(self, dt, x):
        """ eps-greedy policy """
        if random.uniform(0, 1) > self.eps:
            # greedy control/action
            self.u = [self.controls[self.argmin_qNetwork(x)]]
            self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
            self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        else:
            self.u = self.take_random_action(dt)


        return self.u

    def take_random_action(self, dt):
        """ random policy """
        self.u = [random.choice(self.controls)]
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def training_data(self, dataSet):
        batch = dataSet.data
        inputs = torch.randn(len(batch), self.qNetwork.netStructure[0])
        targets = torch.randn(len(batch), self.qNetwork.netStructure[-1])
        for i in range(len(batch)):
            x_ = batch[i]['x_']
            u = batch[i]['u']
            x = batch[i]['x']
            c = batch[i]['c']
            inputs[i, :] = torch.Tensor([x_ + u])
            if c == 0.0 or c == 1.0:
                target = torch.Tensor([[c]])
            else:
                target = torch.Tensor(torch.Tensor([[torch.Tensor([c]) + self.min_qNetwork(x)*self.gamma]]))
            targets[i, :] = target

        return inputs, targets

    def artificial_data(self, n):
        targets_htg = torch.zeros(len(self.controls)*n, self.qNetwork.netStructure[-1])
        inputs = torch.Tensor(len(self.controls), self.qNetwork.netStructure[0])
        for i, u in enumerate(self.controls):
            inputs[i, :] = torch.Tensor([self.xGoal + [u]])
        inputs_htg = inputs
        for _ in range(n-1):
            inputs_htg = torch.cat((inputs_htg, inputs), 0)
        return inputs_htg, targets_htg

    def min_qNetwork(self, x):
        qValues = []
        for u in self.controls:
            qValues.append(self.qNetwork(torch.Tensor(x + [u])))
        return np.min(qValues)

    def argmin_qNetwork(self, x):
        qValues = []
        for u in self.controls:
            qValues.append(self.qNetwork(torch.Tensor(x + [u])))
        return np.argmin(qValues)

class NFQ(Algorithm):
    """ Neural Fitted Q Iteration (NFQ) - Implementation based on PyTorch (https://pytorch.org)

        Riedmiller M. (2005) Neural Fitted Q Iteration – First Experiences with a
        Data Efficient Neural Reinforcement Learning Method.

        In: Gama J., Camacho R., Brazdil P.B., Jorge A.M., Torgo L. (eds)
        Machine Learning: ECML 2005. ECML 2005.
        Lecture Notes in Computer Science, vol 3720. Springer, Berlin, Heidelberg

        DOI: https://doi.org/10.1007/11564096_32

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

    def __init__(self, environment, controls, xGoal, t, dt, h_layers=[20,20], eps=0,
                 gamma=0.99, path='../results/ddpg/', nData=180000):
        self.path = path
        xGoal = observation(xGoal, environment.xIsAngle)
        uDim = 1 # dimension input
        netStructure = [len(xGoal) + uDim] + h_layers + [1]
        agent = QNetwork(controls, netStructure, eps, gamma, xGoal)
        super(NFQ, self).__init__(environment, agent, t, dt)
        self.D = DataSet(nData)
        self.episode = 0
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path + 'plots/'):
            os.makedirs(path + 'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')
        copyfile(inspect.stack()[-1][1], path + 'exec_script.py')
        self.cost_scale = int(1/dt)
        self.environment.terminal_cost = 1.

    def run_episode(self):
        print('started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        # list of incremental costs
        cost = []
        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()
        for _ in tt:
            # agent computes control/action
            if self.episode > 0:
                u = self.agent.take_action(self.dt, self.environment.o)
            else:
                u = self.agent.take_random_action(self.dt)
            # simulation of environment
            c = self.environment.step(u, self.dt)*self.cost_scale
            if c == 1.0:
                self.environment.terminated = True
            elif c == 0.0:
                print('Goal reached at t = ',_)
            cost.append(c)

            # store transition in dataset (x_, u, x, c)
            transition = ({'x_': self.environment.o_, 'u': self.agent.u, 'x': self.environment.o, 'c': c})
            self.D.add_sample(transition)

            if self.environment.terminated:
                print('Environment terminated!')
                break

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        print('Mean cost: ',np.mean(cost))
        # learning
        self.agent.train(self.D)

        self.episode += 1
        pass

    def run_learning(self, n):
        self.episode = 0
        self.meanCost = []
        for k in range(n):
            self.run_episode()
            # plot environment after episode finished
            print('Samples: ',self.D.data.__len__())
            if k % 10 == 0:
                self.learning_curve()
                # if self.meanCost[-1] < 0.01: # goal reached
                #self.animation()
            if k % 50 == 0:
                self.plot()
                self.animation()
        pass

    def plot(self):
        """ Plots the environment's and agent's history. """

        self.environment.plot()
        self.environment.save_history(str(self.episode - 1) + '_environment', self.path + 'data/')
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pdf')
        try:
            plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pgf')
        except:
            pass
        self.agent.plot()
        self.agent.save_history(str(self.episode - 1) + '_agent', self.path + 'data/')
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

        fig, ax = plt.subplots(1, 1, dpi=150, sharex=True, figsize=(5.56, 3.44))

        #x = np.arange(1, self.episode)
        x = np.arange(self.episode)

        ax.step(x, self.meanCost, 'b', lw=1, label=r'$\frac{1}{N}\sum_{k=0}^N c_k$')
        ax.legend(loc='center', bbox_to_anchor=(1.15, .5), ncol=1, shadow=True)
        ax.grid(True)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3,4), useMathText=True)
        plt.rc('font', family='serif')
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.savefig(self.path + 'learning_curve.pdf')
        plt.close('all')
        pass