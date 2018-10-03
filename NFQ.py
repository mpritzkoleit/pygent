import numpy as np
from Agent import Agent
from Data import DataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import abstractmethod
import matplotlib.pyplot as plt
from NeuralNetworkModels import MLP
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

    def __init__(self, controls, netStructure, eps=0.1, gamma=0.99):
        super(QNetwork, self).__init__(1)
        self.controls = controls
        self.qNetwork = MLP(netStructure)
        self.eps = eps
        self.gamma = gamma
        # implement neural network in pytorch


    def train(self, dataSet):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Rprop(self.qNetwork.parameters())


        for epoch in range(200):  # loop over the dataset multiple times
            running_loss = 0.0
            # get training data
            outputs, targets = self.training_data(dataSet)

            # zero the parameter gradients
            optimizer.zero_grad()

            # backward + optimize
            #loss = criterion(outputs, targets)
            loss = torch.sum((outputs-targets)**2)
            # print(loss)
            loss.backward()
            optimizer.step()

        pass

    def take_action(self, dt, x):
        """ eps-greedy policy """
        if random.uniform(0, 1) > self.eps:
            # greedy control/action
            self.u = [self.controls[self.argmin_qNetwork(x)]]
        else:
            self.u = [random.choice(self.controls)]
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time

        return self.u

    def take_random_action(self, dt, x):
        """ random policy """
        self.u = [random.choice(self.controls)]
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time

        return self.u

    def training_data(self, dataSet):
        batch = dataSet.data
        inputs = torch.Tensor(len(batch), self.qNetwork.netStructure[0])
        targets = torch.Tensor(len(batch), self.qNetwork.netStructure[-1])
        for i in range(len(batch)):
            x_ = batch[i]['x_']
            u = batch[i]['u']
            x = batch[i]['x']
            c = batch[i]['c']
            inputs[i, :] = torch.Tensor([x_ + u])
            targ = c + self.gamma * self.min_qNetwork(x)
            targets[i, :] = torch.Tensor(targ)

        outputs = self.qNetwork(inputs)
        return outputs, targets

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

class LearningProcess(object):
    """ Learning Process

    Attributes:
        n (int): number of episodes
        t (int, float): episode length
        dt (int, float): step size
        meanCost (int, float): mean cost of an episode
        agent (Agent(object)): agent of the algorithm
        environment (Environment(object)): environment

    """

    meanCost = []

    def __init__(self, environment, agent, n, t, dt):
        self.n = n
        self.t = t
        self.dt = dt
        self.agent = agent
        self.environment = environment

    @abstractmethod
    def run_episode(self):
        return

    @abstractmethod
    def learning_curve(self):
        return


class NFQ(LearningProcess):
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

    def __init__(self, environment, controls, t, dt, n=200, netStructure=[3, 20, 20, 1], eps=0.1, nData=50000):
        agent = QNetwork(controls, netStructure, eps)
        super(NFQ, self).__init__(environment, agent, n, t, dt)
        self.D = DataSet(nData)
        self.episode = 0

    def run_episode(self):
        print('started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        # reset environment to initial state
        x0 = list(self.environment.history[0])
        # list of incremental costs
        cost = []
        self.environment.reset(x0)
        self.agent.reset()
        for _ in tt:
            # agent computes control/action
            if self.episode > 1:
                u = self.agent.take_action(self.dt, self.environment.x)
            else:
                u = self.agent.take_random_action(self.dt, self.environment.x)
            # simulation of environment
            c = self.environment.step(self.dt, u)

            cost.append(c)

            # store transition in dataset (x_, u, x, c)
            transition = ({'x_': self.environment.x, 'u': self.agent.u, 'x': self.environment.x, 'c': c})
            self.D.add_sample(transition)

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        print('Mean cost: ',np.mean(cost))
        # learning
        self.agent.train(self.D)

        self.episode += 1

        pass

    def run_learning(self, n):
        self.episode = 0
        for k in range(n):
            self.run_episode()
            # plot environment after episode finished
            if k % 10 == 0:
                self.plot()

            if k % 25 == 0 and k != 0:
                self.learning_curve()

    def plot(self):
        plt.close()
        self.environment.plot()
        self.agent.plot()

    def learning_curve(self):
        fig, (ax) = plt.subplots(1, 1)
        ax.plot(np.arange(self.episode), self.meanCost)
        plt.title('Learning curve')
        plt.ylabel('Mean cost')
        plt.show()