import numpy as np
import Agent
from Environment import StateSpaceModel
import torch
from abc import abstractmethod

class QNetwork(Agent):
    """ Q-Network (Multi-Layer-Perceptron)
        Q(x,u) -> R

        mu(x) = argmin_u*(Q(x[k],u*)


    Attributes:
        layers (array): array that describes layer structure (i.e. [1, 10, 10, 1])
        eps (float [0, 1]): with probability eps a random action/control is returned
    """

    def __init__(self, layers, eps):
        self.layers = layers
        self.eps = eps
        # implement neural network in pytorch


class LearningProcess(object):
    """ Learning Process

    Attributes:
        n (int): number of episodes
        t (int, float): episode length
        dt (int, float): step size
        meanCost (int, float): mean cost of an episode

    """

    meanCost = np.array([])

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
    """ Neural Fitted Q Iteration (NFQ) - Implementation based on PyTorch

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
        meanCost (int, float): mean cost of an episode
        self.agent

    """

    def __init__(self, environment, t, dt, n=200, layers=[4, 20, 20, 1], eps=0.1):
        agent = QNetwork(layers, eps)
        super(NFQ, self).__init__(environment, agent, n, t, dt)
        self.

    def run_episode(self):
