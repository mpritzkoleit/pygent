import numpy as np
from Agents import Agent
from Data import DataSet
from Algorithms import Algorithm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NeuralNetworkModels import Actor, Critic  # ActorBN as Actor, CriticBN as Critic #,
import os
import pickle


class DDPG(Algorithm):
    """ Deep Deterministic Policy Gradient - Implementation based on PyTorch (https://pytorch.org)

    Paper: Lillicrap, Timothy P. et al. “Continuous control with deep reinforcement learning.”

    Link: https://arxiv.org/abs/1509.02971

    Attributes:
        xDim (int): state dimension (input dimension of the policy network)
        uDim (int): control/action dimension (output dimension of the policy network)
        uMax (list): control/action limits (uMin = - uMax)
        batch_size (int): size of the minibatch used for training
        agent (Agent):
        a_lr = actor (policy) learning rate
        c_lr = critic (q-function) learning rate
        R (DataSet): data set for storing transition tuples
        plotInterval (int)
        checkInterval (int)
        path (string)
        costScale (int): scale of the cost function (numerical advantage)
        warm_up (int): number of random samples, before training begins

    """

    def __init__(self, environment, t, dt, plotInterval=50, nData=1e6, path='../Results/DDPG/', checkInterval=50,
                 evalPolicyInterval=100, costScale=100, warm_up=5e4, a_lr=1e-4, c_lr=1e-3, tau=0.001, batch_size=64):
        xDim = environment.oDim
        uDim = environment.uDim
        uMax = environment.uMax
        self.batch_size = batch_size
        agent = ActorCritic(xDim, uDim, torch.Tensor(uMax), dt, batch_size=self.batch_size, actor_lr=a_lr,
                            critic_lr=c_lr, tau=tau)
        super(DDPG, self).__init__(environment, agent, t, dt)
        self.R = DataSet(nData)
        self.plotInterval = plotInterval  # inter
        self.evalPolicyInterval = evalPolicyInterval
        self.checkInterval = checkInterval  # checkpoint interval
        self.path = path
        self.costScale = costScale
        self.warm_up = warm_up
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path + 'plots/'):
            os.makedirs(path + 'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')

    def run_episode(self):
        """ Run a training episode. If terminal state is reached, episode stops."""

        print('Started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        cost = []  # list of incremental costs

        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()

        for _ in tt:
            # agent computes control/action
            if self.episode % self.evalPolicyInterval == 0 and self.R.data.__len__() >= self.warm_up:
                u = self.agent.take_action(self.dt, self.environment.o)
            else:
                u = self.agent.take_random_action(self.dt, self.environment.o)
            # simulation of environment
            c = self.environment.step(self.dt, u) * self.costScale
            cost.append(c)

            # store transition in data set (x_, u, x, c)
            transition = ({'x_': self.environment.o_, 'u': self.agent.u, 'x': self.environment.o, 'c': [c]})

            # add sample to data set
            self.R.force_add_sample(transition)

            # training of the policy network (agent)
            if self.R.data.__len__() > self.warm_up:
                self.agent.training(self.R)

            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        self.totalCost.append(np.sum(cost))
        # todo: add expected cost
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

        for _ in tt:
            # agent computes control/action
            u = self.agent.take_action(self.dt, self.environment.o)

            # simulation of environment
            c = self.environment.step(self.dt, u)
            cost.append(c)

            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        self.totalCost.append(np.sum(cost))
        self.episode += 1
        pass

    def run_learning(self, n):
        """ Learning process.

            Args:
                n (int): number of episodes
        """

        for k in range(1, n + 1):
            self.run_episode()
            # plot environment after episode finished
            print('Samples: ', self.R.data.__len__())
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
        torch.save({'actor': self.agent.actor1.state_dict(),
                    'critic': self.agent.critic1.state_dict()}, self.path + 'data/checkpoint.pth')

        # save data set
        self.R.save(self.path + 'data/dataSet.p')

        # save learning curve data
        pickle.dump(self.meanCost, open(self.path + 'data/meanCost.p', 'wb'))
        pickle.dump(self.totalCost, open(self.path + 'data/totalCost.p', 'wb'))
        pass

    def load(self):
        """ Load neural network parameters and data set. """

        # load network parameters
        if os.path.isfile(self.path + 'data/checkpoint.pth'):
            checkpoint = torch.load(self.path + 'data/checkpoint.pth')
            self.agent.actor1.load_state_dict(checkpoint['actor'])
            self.agent.actor2.load_state_dict(checkpoint['actor'])
            self.agent.critic1.load_state_dict(checkpoint['critic'])
            self.agent.critic2.load_state_dict(checkpoint['critic'])
        else:
            print('Checkpoint file not found!')

        # load data set
        if os.path.isfile(self.path + 'data/dataSet.p'):
            self.R.load(self.path + 'data/dataSet.p')
        else:
            print('No dataset found!')

        # load learning curve
        if os.path.isfile(self.path + 'data/meanCost.p'):
            self.meanCost = pickle.load(open(self.path + 'data/meanCost.p', 'rb'))
            self.episode = self.meanCost.__len__() + 1
        else:
            print('No learning curve data found!')
        if os.path.isfile(self.path + 'data/totalCost.p'):
            self.totalCost = pickle.load(open(self.path + 'data/totalCost.p', 'rb'))
        else:
            print('No learning curve data found!')
        pass

    def plot(self):
        """ Plots the environment's and agent's history. """

        self.environment.plot()
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pdf')
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pgf')
        self.agent.plot()
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_agent.pdf')
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_agent.pgf')
        plt.close('all')
        pass

    def animation(self):
        """ Animation of the environment (if available). """

        ani = self.environment.animation()
        if ani != None:
            ani.save(self.path + 'animations/' + str(self.episode - 1) + '_animation.mp4', fps=1 / self.dt)
        plt.close('all')
        pass

    def learning_curve(self):
        """ Plot of the learning curve. """

        fig, ax = plt.subplots(2, 1, dpi=150, sharex=True)
        ax[0].step(np.arange(1, self.episode), self.meanCost, 'b', lw=1)
        ax[0].set_ylabel(r'avg. cost/step')
        ax[0].grid(True)
        ax[1].step(np.arange(1, self.episode), self.totalCost, 'b', lw=1)
        ax[1].set_ylabel(r'total cost')
        ax[1].grid(True)
        plt.xlabel(r'Episode')
        plt.savefig(self.path + 'learning_curve.pdf')
        plt.savefig(self.path + 'learning_curve.pgf')
        # todo: save learning curve data
        # todo: plot expected return
        plt.close('all')
        pass


class ActorCritic(Agent):
    """ Actor-Critic agent. (Specialized for the DDPG algorithm.)

    Critic: Q(x,u), Q-network (multi-layer-perceptron)

    Actor: mu(x), policy network (multi-layer-perceptron)


        Attributes:
            xDim (int): state dimension (input dimension of the policy network)
            uDim (int): control/action dimension (output dimension of the policy network)
            uMax (list): control/action limits (uMin = - uMax)
            actor1 (Actor): policy network for training
            actor2 (Actor): policy network for targets
            critic1 (Critic): Q-network for training
            critic2 (Critic): Q-network for targets
            gamma (float): discount factor
            tau (float): blend factor
            optimCritic (torch.optim.Adam): optimizer for the critic (Q-network)
            optimActor (torch.optim.Adam): optimizer for the actor (policy network)
            noise (OUnoise): noise process for control/action noise
            batch_size (int): size of the minibatch used for training

    """

    def __init__(self, xDim, uDim, uMax, dt, batch_size=128, gamma=0.99, tau=0.001, actor_lr=1e-4, critic_lr=1e-3):
        super(ActorCritic, self).__init__(uDim)
        self.xDim = xDim
        self.uMax = uMax
        self.actor1 = Actor(xDim, uDim, uMax)
        self.actor2 = Actor(xDim, uDim, uMax)
        self.blend_hard(self.actor1, self.actor2)  # equate parameters of actor networks
        self.critic1 = Critic(xDim, uDim)
        self.critic2 = Critic(xDim, uDim)
        self.blend_hard(self.critic1, self.critic2)  # equate parameters of critic networks
        self.gamma = gamma  # discount factor
        self.tau = tau  # blend factor
        self.optimCritic = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=1e-2)
        self.optimActor = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr)
        self.noise = OUnoise(uDim, dt)  # exploration noise
        self.batch_size = batch_size

    def training(self, dataSet):
        """ Training of the Q-network and policy network.

            Args:
                dataSet (DataSet): data set of transition tuples
        """

        # loss function (mean squared error)
        criterion = nn.MSELoss()

        # create training data/targets
        x_Inputs, uInputs, qTargets = self.training_data(dataSet)

        for epoch in range(1):  # loop over the dataset multiple times
            # output of the Q-network
            qOutputs = self.critic1(x_Inputs, uInputs)
            qOutputs = torch.squeeze(qOutputs)

            # output of the policy network
            muOutputs = self.actor1(x_Inputs)

            self.train() # train mode on (batch normalization)

            # definition of loss functions
            lossCritic = criterion(qOutputs, qTargets)
            lossActor = self.critic1(x_Inputs,  muOutputs).mean()  # *-1 when using rewards instead of costs

            # train Q-network
            self.optimCritic.zero_grad()  # delete gradients
            lossCritic.backward()  # error back-propagation
            self.optimCritic.step()  # gradient descent step

            # train policy network
            self.optimActor.zero_grad()  # delete gradients
            lossActor.backward()  # error back-propagation
            self.optimActor.step()  # gradient descent step

            # blend target networks
            self.blend(self.critic1, self.critic2)
            self.blend(self.actor1, self.actor2)

            self.eval() # eval mode on (batch normalization)
        pass

    def train(self):
        """ Set Q-networks and policy networks to 'train' mode.
        Only needed, when networks have a batch normalization layer. """

        self.actor1.train()
        self.actor2.train()
        self.critic1.train()
        self.critic2.train()
        pass

    def eval(self):
        """ Set Q-networks and policy networks to 'eval' mode.
            Only needed, when networks have a batch normalization layer. """

        self.actor1.eval()
        self.actor2.eval()
        self.critic1.eval()
        self.critic2.eval()
        pass

    def blend(self, source, target):
        """ Blend parameters of a target neural network with parameters from a source network.

            Args:
                source (torch.nn.Module): source neural network
                target (torch.nn.Module): target neural network
        """

        for wTarget, wSource in zip(target.parameters(), source.parameters()):
            wTarget.data.copy_(self.tau * wSource.data + (1.0 - self.tau) * wTarget.data)
        pass

    def blend_hard(self, source, target):
        """ Copy parameters from one neural network to another.

                    Args:
                        source (torch.nn.Module): source neural network
                        target (torch.nn.Module): target neural network
        """

        for wTarget, wSource in zip(target.parameters(), source.parameters()):
            wTarget.data.copy_(wSource.data)
        pass

    def take_action(self, dt, x):
        """ Compute the control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): control/action
        """

        self.eval()
        x = torch.Tensor([x])
        self.u = np.asarray(self.actor1(x).detach())[0]
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def take_random_action(self, dt, x):
        """ Compute the noisy control/action of the policy network (actor).

            Args:
                dt (float): stepsize
                x (ndarray, list): state (input of policy network)

            Returns:
                u (ndarray): noisy control/action
        """

        self.eval()
        x = torch.Tensor([x])
        u = np.asarray(self.actor1(x).detach())[0] + self.noise.sample()*self.uMax.numpy()
        self.u = np.clip(u, -self.uMax.numpy(), self.uMax.numpy())
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def training_data(self, dataSet):
        """ Create training data for the critc (Q-network).

            Args:
                dataSet (DataSet): data set with transition tuples

            Returns:
                x_Inputs (torch.Tensor): state tensor
                uInputs (torch.Tensor): control/action tensor
                qTargets (torch.Tensor): target value tensor for the Q-network
        """

        batch = dataSet.minibatch(self.batch_size)
        x_Inputs = torch.Tensor([sample['x_'] for sample in batch])
        xInputs = torch.Tensor([sample['x'] for sample in batch])
        uInputs = torch.Tensor([sample['u'] for sample in batch])
        costs = torch.Tensor([sample['c'] for sample in batch])

        self.eval()  # evaluation mode (for batch normalization)
        qTargets = costs + self.gamma * self.critic2(xInputs, self.actor2(xInputs)).detach()
        qTargets = torch.squeeze(qTargets)
        return x_Inputs, uInputs, qTargets


class OUnoise(object):
    """ Ornstein-Uhlenbeck process. (discrete Euler approximation)

    Implementation based on:
    https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

        Args:
            uDim (int): control/action dimensions
            mu (float): expected value
            theta (float): stiffness parameter
            sigma (float): diffusion parameter
            x (ndarray): current state
            dt (float): step size

    """

    # Todo: move to Utilities.py
    def __init__(self, uDim, dt, mu=0, theta=0.15, sigma=0.2):
        self.uDim = uDim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones(self.uDim) * self.mu
        self.dt = dt

    def reset(self):
        """ Reset process state to mean value. """
        self.x = np.ones(self.uDim) * self.mu

    def sample(self):

        dx = self.theta*(self.mu - self.x)*self.dt + self.sigma*np.random.normal(0, self.dt, len(self.x))
        self.x = self.x + dx
        return self.x
