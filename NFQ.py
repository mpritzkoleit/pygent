import numpy as np
from Agents import Agent
from Data import DataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import abstractmethod
import matplotlib.pyplot as plt
from NeuralNetworkModels import MLP
from Algorithms import Algorithm
import random

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

    def plot(self):
        """ Plots the agents history

        Returns:
            fig (matplotlib.pyplot.figure)

        """

        fig, (ax) = plt.subplots(1, 1)
        for i in range(len(self.u)):
            ax.step(self.tt, self.history[:, i], label=r'$u_'+str(i+1)+'$')
        ax.grid(True)
        plt.xlabel('t in s')
        plt.title('Controls')
        ax.legend()

        return fig

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

    def __init__(self, environment, controls, xGoal, t, dt, netStructure, eps, gamma, nData=180000):
        agent = QNetwork(controls, netStructure, eps, gamma, xGoal)
        super(NFQ, self).__init__(environment, agent, t, dt)
        self.D = DataSet(nData)
        self.episode = 0

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
                u = self.agent.take_action(self.dt, self.environment.x)
            else:
                u = self.agent.take_random_action(self.dt)
            # simulation of environment
            c = self.environment.step(self.dt, u)
            if c == 1.0:
                self.environment.terminated = True
            elif c == 0.0:
                print('Goal reached at t = ',_)
            cost.append(c)

            # store transition in dataset (x_, u, x, c)
            transition = ({'x_': self.environment.x_, 'u': self.agent.u, 'x': self.environment.x, 'c': c})
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
                self.plot()
                self.learning_curve()
                # if self.meanCost[-1] < 0.01: # goal reached
                #self.animation()


    def plot(self):
        self.environment.plot()
        plt.savefig('results/'+str(self.episode-1)+'_environment')
        self.agent.plot()
        plt.savefig('results/'+str(self.episode-1)+'_agent')
        plt.close('all')

    def animation(self):
        ani = self.environment.animation(self.episode-1, self.meanCost[self.episode-1])
        ani.save('results/'+str(self.episode-1)+'_animation.mp4', fps=1/self.dt)
        plt.close('all')


    def learning_curve(self):
        fig, (ax) = plt.subplots(1, 1)
        ax.step(np.arange(self.episode), self.meanCost)
        plt.title('Learning curve')
        plt.ylabel('Mean cost')
        plt.xlabel('Epsiode')
        plt.savefig('results/learning_curve')
        plt.close('all')
