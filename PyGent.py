import numpy as np
#import tensorflow as tf
import abc
import scipy.integrate as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch

class Environment(object):
    """ Base class for an environment.

    Attributes:
        x (int, float, ndarray): current state x[k] (size = n)
        x_ (int, float, ndarray): previous state x[k-1](size = n)
        history (ndarray)
        terminated (bool)

    """
    tt = [0]
    terminated = False  # variable that stores, if a final state is reached

    def __init__(self, x0):
        self.x = np.array(x0) # current state
        self.x_ = np.array(x0) # previous state x[k-1]
        self.history = np.array([x0])

    def get_state(self):
        return self.x

    def reset(self, x0):
        self.history = np.array([x0])
        self.x_ = np.array(x0)
        self.x = np.array(x0)

    @abc.abstractmethod
    def step(self, *args):
        return

    @abc.abstractmethod
    def cost(self, *args):
        return

    @abc.abstractmethod
    def plot(self, *args):
        return

    def animation(self, *args):
        return



class StateSpaceModel(Environment):
    """Environment subclass that uses a state space model of the form dx/dt = f(x,u)

    Attributes:
        ode (function): ODE for simulation
        ode (function): ODE for simulation

    """

    def __init__(self, ode, x0):
        super(StateSpaceModel, self).__init__(x0)
        self.ode = ode

    def get_cost(self, x, u):
        c = 0
        return c

    def step(self, t, u):
        """Simulation of the environment for 1 step of time t.

        Args:
            t (int, float): simulation time
            u (ndarray): control/action

        Returns:
            x (ndarray): new current state
        """
        # system simulation
        sol = solve_ivp(lambda t, x: self.ode(t, x, u), (0, t), self.x)

        # step
        self.x_ = self.x # shift state
        self.x = sol.y[:, -1]  # extract simulation result
        self.history = np.concatenate((self.history, np.array([self.x]))) # save current state
        self.tt.extend([self.tt[-1] + t]) # increment simulation time
        #c = self.get_cost(self.x, u)

        return self.x

    def plot(self):
        fig, (ax) = plt.subplots(1, 1)
        for i in range(len(self.x)):
            ax.plot(self.tt, self.history[:,i], label=r'$x_'+str(i+1)+'$')
        ax.grid(True)
        plt.xlabel('t in s')
        ax.legend()
        return fig

class Agent(object):
    """ Base class for an agent. """

    @abc.abstractmethod
    def take_action(self, *args):
        return

class FeedBack(Agent):
    """Environment subclass that represents a standard state feedback u = mu(x)

        Attributes:
            mu (function): feedback law

        """
    def __init__(self, mu, m):
        super(FeedBack, self).__init__()
        self.mu = mu
        self.tt = [0]
        self.history = np.zeros([1, m])
    def take_action(self, t, x):
        self.u = self.mu(x)
        self.history = np.concatenate((self.history, np.array([self.u])))
        self.tt.extend([self.tt[-1] + t])  # increment simulation time
        return self.u

    def plot(self):
        fig, (ax) = plt.subplots(1, 1)
        for i in range(len(self.u)):
            ax.plot(self.tt, self.history[:, i],label=r'$u_'+str(i+1)+'$')
        ax.grid(True)
        plt.xlabel('t in s')
        ax.legend()
        return fig

'''
class NFQ(Agent):
    """ Neural Fitted Q-Iteration """
    dataset

class DDPG(Agent):
    """ Deep Deterministic Policy Gradient """    

class DQN(Agent):
    """ Deep Q-Network """   

class REINFORCE(Agent):
    """ Deep Distributed Distributional Deterministic Policy Gradient """   

class VanillaPG(Agent):
    """ Vanilla Policy Gradient """  
    
    
class D4PG(Agent):
    """ Deep Distributed Distributional Deterministic Policy Gradient """   
       
class DataSet(object):
    random_sample
    append
    minibatch
    size
    
class MLP(object):
    layer_structure [1, 10, 10]
    weights
    train (input, output)
    test (learning_data, test_data)
    feedforward_pass
    train(steps)
    loss_function
    training_curve
    
class Actor(MLP)
class Critic(MLP)

class LearningSystem(object)
    num_Episodes
    time_steps
    dt
    show_learning progress

class OpenAIGymEnv(Environment):


'''


