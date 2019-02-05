import numpy as np
from Agents import Agent
from Data import DataSet
from Algorithms import Algorithm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NeuralNetworkModels import Actor, Critic# ActorBN as Actor, CriticBN as Critic #,
import os
import pickle
import random
import time

class DDPG(Algorithm):
    """ Deep Deterministic Policy Gradient - Implementation based on PyTorch (https://pytorch.org)

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

    def __init__(self, environment, t, dt, plotInterval=10, nData=1e6, path='../Controllers/DDPG/', checkInterval=100, evalPolicyInterval=50):
        xDim = environment.oDim
        uDim = environment.uDim
        uMax = environment.uMax
        agent = ActorCritic(xDim, uDim, torch.Tensor(uMax), dt)
        super(DDPG, self).__init__(environment, agent, t, dt)
        self.R = DataSet(nData)
        self.plotInterval = plotInterval # inter
        self.evalPolicyInterval = evalPolicyInterval
        self.checkInterval = checkInterval  # checkpoint interval
        self.epsDecay = 1.
        self.path = path
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path+'plots/'):
            os.makedirs(path+'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')

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
            if self.episode % self.evalPolicyInterval == 0:
                u = self.agent.take_action(self.dt, self.environment.o)
            else:
                u = self.agent.take_random_action(self.dt, self.environment.o)
            # simulation of environment
            c = self.environment.step(self.dt, u)
            cost.append(c)

            # store transition in dataset (x_, u, x, c)
            transition = ({'x_': self.environment.o_, 'u': self.agent.u, 'x': self.environment.o, 'c': [c]})
            self.R.add_sample(transition)
            if self.R.data.__len__() > 64 and self.episode > 0:
                self.agent.training(self.R)
            if self.environment.terminated:
                print('Environment terminated!')
                break
        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        self.episode += 1

        pass

    def run_controller(self, x0):
        print('started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        # list of incremental costs
        cost = []
        # reset environment/agent to initial state, delete history
        self.environment.reset(x0)
        self.agent.reset()
        for _ in tt:
            # agent computes control/action
            u = self.agent.take_action(self.dt, self.environment.o)

            # simulation of environment
            c = self.environment.step(self.dt, u)
            cost.append(c)

            if self.environment.terminated:
                print('Environment terminated!')
                break
        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        pass

    def run_learning(self, n):
        for k in range(1, n+1):
            self.run_episode()
            # plot environment after episode finished
            print('Samples: ',self.R.data.__len__())
            if k % 10 == 0:
                self.learning_curve()
            if k % self.checkInterval == 0:
                self.save()
                # if self.meanCost[-1] < 0.01: # goal reached
            if k % self.plotInterval == 0:
                self.plot()
                self.animation()

    def save(self):
        # save model
        torch.save({'actor': self.agent.actor1.state_dict(),
                 'critic': self.agent.critic1.state_dict()}, self.path+'data/checkpoint.pth')
        self.R.save(self.path+'data/dataSet.p')
        pickle.dump(self.meanCost, open(self.path+'data/learningCurve.p', 'wb'))
        pass

    def load(self):
        if os.path.isfile(self.path+'data/checkpoint.pth'):
            checkpoint = torch.load(self.path+'data/checkpoint.pth')
            self.agent.actor1.load_state_dict(checkpoint['actor'])
            self.agent.actor2.load_state_dict(checkpoint['actor'])
            self.agent.critic1.load_state_dict(checkpoint['critic'])
            self.agent.critic2.load_state_dict(checkpoint['critic'])
        else:
            print('Checkpoint file not found!')
        if os.path.isfile(self.path + 'data/dataSet.p'):
            self.R.load(self.path+'data/dataSet.p')
        else:
            print('No dataset found!')
        if os.path.isfile(self.path + 'data/learningCurve.p'):
            self.meanCost = pickle.load(open(self.path + 'data/learningCurve.p', 'rb'))
            self.episode = self.meanCost.__len__() + 1
        else:
            print('No learning curve data found!')
        pass


    def plot(self):
        self.environment.plot()
        plt.savefig(self.path+'plots/'+str(self.episode-1)+'_environment.pdf')
        self.agent.plot()
        plt.savefig(self.path+'plots/'+str(self.episode-1)+'_agent.pdf')
        plt.close('all')


    def animation(self):
        ani = self.environment.animation(self.episode-1, self.meanCost[self.episode-2])
        if ani != None:
            ani.save(self.path+'animations/'+str(self.episode-1)+'_animation.mp4', fps=1/self.dt)
        plt.close('all')


    def learning_curve(self):
        fig, (ax) = plt.subplots(1, 1, dpi=150)
        ax.step(np.arange(1, self.episode), self.meanCost)
        plt.title('Learning curve')
        plt.ylabel('Mean cost')
        plt.xlabel('Episode')
        plt.savefig(self.path+'learning_curve.pdf')
        plt.close('all')

class ActorCritic(Agent):
    """ Q-Network (Multi-Layer-Perceptron)
        Q(x,u) -> R

        mu(x) = argmin_u*(Q(x[k],u*)


    Attributes:
        controls (array): control/actions that the agent can choose from
        netStructure (array): array that describes layer structure (i.e. [1, 10, 10, 1])
        eps (float [0, 1]): with probability eps a random action/control is returned
    """

    def __init__(self, xDim, uDim, uMax, dt, gamma=0.99, tau=0.001):
        super(ActorCritic, self).__init__(1)
        self.xDim = xDim
        self.uDim = uDim
        self.uMax = uMax
        self.actor1 = Actor(xDim, uDim, uMax)
        self.actor2 = Actor(xDim, uDim, uMax)
        self.blend_hard(self.actor1, self.actor2)
        self.critic1 = Critic(xDim, uDim)
        self.critic2 = Critic(xDim, uDim)
        self.blend_hard(self.critic1, self.critic2)
        self.gamma = gamma
        self.tau = tau
        self.optimCritic = torch.optim.Adam(self.critic1.parameters(), lr=1e-3, weight_decay=1e-2)
        self.optimActor = torch.optim.Adam(self.actor1.parameters(), lr=1e-4)
        self.noise = OUnoise(uDim, dt)

    def training(self, dataSet):
        # loss function (mean squared error)
        criterion = nn.MSELoss()

        # training data
        x_Inputs, uInputs, qTargets = self.training_data(dataSet)
        # eval mode on (batch normalization)
        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0
            qOutputs = self.critic1(x_Inputs, uInputs)
            qOutputs = torch.squeeze(qOutputs)
            muOutputs = self.actor1(x_Inputs)
            self.train()
            # backward + optimize
            lossCritic = criterion(qOutputs, qTargets)
            lossActor = self.critic1(x_Inputs, muOutputs).mean() # -self.critic1(xInputs, muOutputs).mean() when using rewards instead of cost

            #loss.backward(retain_graph=True)
            self.optimCritic.zero_grad()
            lossCritic.backward()
            self.optimCritic.step()

            self.optimActor.zero_grad()
            lossActor.backward()
            self.optimActor.step()

            self.blend(self.critic1, self.critic2)
            self.blend(self.actor1, self.actor2)
            # eval mode on (batch normalization)
            self.eval()
        pass

    def train(self):
        self.actor1.train()
        self.actor2.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor1.eval()
        self.actor2.eval()
        self.critic1.eval()
        self.critic2.eval()

    def blend(self, source, target):
        for wTarget, wSource in zip(target.parameters(), source.parameters()):
            wTarget.data.copy_(self.tau*wSource.data + (1.0 - self.tau)*wTarget.data)
        return

    def blend_hard(self, source, target):
        for wTarget, wSource in zip(target.parameters(), source.parameters()):
            wTarget.data.copy_(wSource.data)
        return

    def take_action(self, dt, x):
        """ eps-greedy policy """
        # greedy control/action
        self.eval()
        x = torch.Tensor([x])
        self.u = np.asarray(self.actor1(x).detach())[0]
        # self.u = np.clip(np.asarray(self.actor1(x).detach())[0] + self.noise()*self.uMax, -self.uMax, self.uMax)
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def take_random_action(self, dt, x):
        """ random policy """
        #self.u = [np.random.uniform(-self.uMax, self.uMax)]
        self.eval()
        x = torch.Tensor([x])
        u = np.asarray(self.actor1(x).detach())[0] + self.noise.sample()*self.uMax.numpy()
        self.u = np.clip(u, -self.uMax.numpy(), self.uMax.numpy())
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def training_data(self, dataSet):
        # eval mode on (batch normalization)
        batch = dataSet.minibatch(64)
        x_Inputs = torch.Tensor([sample['x_'] for sample in batch])
        xInputs =  torch.Tensor([sample['x'] for sample in batch])
        uInputs =  torch.Tensor([sample['u'] for sample in batch])
        costs =  torch.Tensor([sample['c'] for sample in batch])
        self.eval()
        qTargets = costs + self.gamma*self.critic2(xInputs, self.actor2(xInputs)).detach()
        qTargets = torch.squeeze(qTargets)
        return x_Inputs, uInputs, qTargets

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

class OUnoise:
    def __init__(self, action_dim, dt, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu
        self.dt =dt

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)*self.dt + self.sigma * np.random.normal(0, self.dt, len(self.X))
        self.X = self.X + dx

        return self.X