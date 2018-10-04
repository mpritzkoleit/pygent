import numpy as np
# import tensorflow as tf
from abc import abstractmethod, abstractproperty
# import scipy.integrate as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
# import torch


class Environment(object):
    """ Base class for an environment.

    Attributes:

        x (array): current state x[k] (size = n)
        x_ (array): previous state x[k-1](size = n)
        history (array): previous states (x[0],x[1],...,x[k-1])
        tt (list): time vector (corresponding to history)
        terminated (bool): True, if environment is in a terminal state

    """

    tt = [0]
    terminated = False

    def __init__(self, x0):
        x0 = list(x0)
        self.x = x0  # current state
        self.x_ = x0 # previous state x[k-1]
        self.history = np.array([x0])

    def get_state(self):
        return self.x

    def reset(self, x0):
        """ Resets environment to state x0

        Args:
            x0 (list): initial state

        Returns:
            None

        """
        x0 = list(x0)
        self.history = np.array([x0])
        self.x_ = x0
        self.x = x0
        self.tt = [0]
        self.terminated = False

        return None

    @abstractmethod
    def step(self, *args):
        return

    @abstractmethod
    def plot(self, *args):
        return

    def animation(self, *args):
        return


class StateSpaceModel(Environment):
    """Environment subclass that uses a state space model of the form dx/dt = f(x,u)

    Attributes:
        ode (function): ODE for simulation
        cost (function): ODE for simulation

    """

    def __init__(self, ode, cost, x0):
        super(StateSpaceModel, self).__init__(x0)
        self.ode = ode
        self.cost = cost
        self.xIsAngle = np.zeros([len(x0)], dtype=bool)

    def step(self, dt, u):
        """ Simulates the environment for 1 step of time t.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (array): control/action

        Returns:
            c (float): cost of state transition

        """

        # system simulation
        sol = solve_ivp(lambda t, x: self.ode(t, x, u), (0, dt), self.x)

        self.x_ = self.x  # shift state (x[k-1] = x[k])
        y = list(sol.y[:, -1])  # extract simulation result
        self.x = self.mapAngles(y)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        c = self.cost(self.x_, u, self.x)
        if c == 1:
            self.terminated = True
        return c

    def mapAngles(self, y):
        x = y
        for i in range(len(y)):
            if self.xIsAngle[i]:
                # map theta to [-pi,pi]
                if x[i] > np.pi:
                    x[i] -= 2 * np.pi
                elif x[i] < -np.pi:
                    x[i] += 2 * np.pi

        return x

    def plot(self):
        """ Plots the environments history

        Returns:
            fig (matplotlib.pyplot.figure)

        """

        fig, (ax) = plt.subplots(1, 1)
        # Plot state trajectories
        for i in range(len(self.x)):
            ax.plot(self.tt, self.history[:, i], label=r'$x_'+str(i+1)+'$')
        ax.grid(True)
        plt.title('States')
        plt.xlabel('t in s')
        ax.legend()

        return fig