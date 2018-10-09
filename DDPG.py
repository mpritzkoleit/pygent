import numpy as np
from Agents import Agent
from Data import DataSet
from Algorithms import Algorithm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NeuralNetworkModels import Actor, Critic
import random

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

    def __init__(self, environment, xDim, uDim, uMax, t, dt, nData=10e6):
        agent = ActorCritic(xDim, uDim, uMax)
        super(DDPG, self).__init__(environment, agent, t, dt)
        self.R = DataSet(nData)
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

            cost.append(c)

            # store transition in dataset (x_, u, x, c)
            transition = ({'x_': self.environment.x_, 'u': self.agent.u, 'x': self.environment.x, 'c': c})
            self.R.add_sample(transition)

            self.agent.training(self.R)

            if self.environment.terminated:
                print('Environment terminated!')
                break

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        print('Mean cost: ',np.mean(cost))
        # learning
        self.agent.training(self.R)

        self.episode += 1

        pass

    def run_learning(self, n):
        self.episode = 0
        self.meanCost = []
        for k in range(n):
            self.run_episode()
            # plot environment after episode finished
            print('Samples: ',self.R.data.__len__())
            if k % 10 == 0:
                self.plot()
                self.learning_curve()
                # if self.meanCost[-1] < 0.01: # goal reached
            if k % 10 == 0:
                self.animation()


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
        plt.xlabel('Episode')
        plt.savefig('results/learning_curve')
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

    def __init__(self, xDim, uDim, uMax, eps=0.01, gamma=0.99, tau=0.001):
        super(ActorCritic, self).__init__(1)
        self.xDim = xDim
        self.uDim = uDim
        self.uMax = uMax
        self.actor1 = Actor(xDim, uDim, uMax)
        self.actor2 = self.actor1
        self.critic1 = Critic(xDim, uDim)
        self.critic2 = self.critic1
        self.eps = eps
        self.gamma = gamma
        self.tau = tau
        self.optimCritic = torch.optim.Adam(self.critic1.parameters(), lr=10e-3)
        self.optimActor = torch.optim.Adam(self.actor1.parameters(), lr=10e-4)


    def training(self, dataSet):
        # loss function (mean squared error)
        criterion = nn.MSELoss()

        # training data
        xInputs, uInputs, qTargets = self.training_data(dataSet)

        # eval mode on (batch normalization)
        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0

            qOutputs = self.critic1(xInputs, uInputs)
            muOutputs = self.actor1(xInputs)

            # backward + optimize
            lossCritic = criterion(qOutputs, qTargets.detach())
            lossActor = -torch.mean(self.critic1(xInputs, muOutputs))

            # zero the parameter gradients
            self.optimCritic.zero_grad()
            self.optimActor.zero_grad()

            #loss.backward(retain_graph=True)
            lossCritic.backward()
            lossActor.backward()

            self.optimCritic.step()
            self.optimActor.step()

            self.blend(self.critic1, self.critic2)
            self.blend(self.actor1, self.actor2)

            # eval mode on (batch normalization)
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
            wTarget.data = self.tau*wSource.data + (1 - self.tau)*wTarget.data
        return

    def take_action(self, dt, x):
        """ eps-greedy policy """
        if random.uniform(0, 1) > self.eps:
            # greedy control/action
            x = torch.Tensor([x])
            self.u = np.asarray(self.actor1(x).detach())[0]
            self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
            self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        else:
            self.take_random_action(dt)

        return self.u

    def take_random_action(self, dt):
        """ random policy """
        self.u = [np.random.uniform(-self.uMax, self.uMax)]
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def training_data(self, dataSet):
        # eval mode on (batch normalization)
        self.actor1.eval()
        self.actor2.eval()
        self.critic1.eval()
        self.critic2.eval()

        batch = dataSet.minibatch(5)
        xInputs = torch.Tensor(len(batch), self.xDim)
        uInputs = torch.Tensor(len(batch), self.uDim)
        qTargets = torch.Tensor(len(batch), 1)
        for i in range(len(batch)):
            x_ = torch.Tensor(batch[i]['x_'])
            u = torch.Tensor(batch[i]['u'])
            x = torch.Tensor([batch[i]['x']])
            c = torch.Tensor([batch[i]['c']])
            xInputs[i, :] = x_
            uInputs[i, :] = u
            qTargets[i, :] = c + self.gamma*self.critic2(x, self.actor2(x))

        return xInputs, uInputs, qTargets

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
