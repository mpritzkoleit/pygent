import numpy as np
# import tensorflow as tf
from abc import abstractmethod, abstractproperty
# import scipy.integrate as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
from matplotlib import animation
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

    def __init__(self, x0):
        if callable(x0):
            self.x0 = x0  # initial state
            x0 = x0()
        else:
            x0 = list(x0)
            self.x0 = x0
        self.x = x0  # current state
        self.x_ = x0 # previous state x[k-1]
        self.n = len(x0)
        self.history = np.array([x0])
        self.tt = [0]
        self.terminated = False

    def get_state(self):
        return self.x

    def reset(self, x0):
        """ Resets environment to state x0

        Args:
            x0 (list): initial state

        Returns:
            None

        """
        if callable(x0):
            x0 = x0()
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

    def animation(self, episode, meanCost):
        return



class StateSpaceModel(Environment):
    """Environment subclass that uses a state space model of the form dx/dt = f(x,u)

    Attributes:
        ode (function): ODE for simulation
        cost (function): ODE for simulation

    """

    def __init__(self, ode, cost, x0):
        super(StateSpaceModel, self).__init__(x0)
        self.n = len(self.x_)
        self.ode = ode
        self.cost = cost
        self.xIsAngle = np.zeros([len(self.x_)], dtype=bool)

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
        c, terminate = self.cost(self.x_, u, self.x)
        self.terminated = terminate
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
            ax (matploltib.pyplot.axes)

        """

        fig, ax = plt.subplots(len(self.x), 1, sharex='col')
        # Plot state trajectories
        fig.suptitle('States')
        for i in range(len(self.x)):
            ax[i].step(self.tt, self.history[:, i], label=r'$x_'+str(i+1)+'$')
            ax[i].grid(True)
            ax[i].legend(loc='upper right')
        plt.xlabel('t in s')

        return fig, ax


class Pendulum(StateSpaceModel):

    def __init__(self, cost, x0):
        super(Pendulum, self).__init__(self.ode, cost, x0)
        self.xIsAngle = [True, False]

    @staticmethod
    def ode(t, x, u):

        g = 9.81  # gravity
        b = 0.1  # dissipation
        u1 = u[0]  # torque
        x1, x2 = x

        dx1dt = x2
        dx2dt = u1 + g * np.sin(x1) - b * x2

        return [dx1dt, dx2dt]

    def animation(self, episode, meanCost):
    # line and text
        def animate(t):
            t = int(t)
            thisx = [0, x[t]]
            thisy = [0, y[t]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % self.tt[t])
            return line, time_text  #

        # mapping from theta to the x,y-plane
        def pendulum_plot(l, theta):
            x = l * np.sin(theta)
            y = l * np.cos(theta)
            return x, y

        [x, y] = pendulum_plot(1, self.history[:, 0])
        fig, (ax) = plt.subplots(1, 1)
        plt.title('Episode ' + str(episode) + ', mean cost: ' + str(meanCost))
        ax.set_aspect('equal')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)),
                                      interval=self.tt[1]*1000, blit=False)
        return ani

class CartPole(StateSpaceModel):
    def __init__(self, cost, x0):
        super(CartPole, self).__init__(self.ode, cost, x0)
        self.xIsAngle = [False, True, False, False]

    @staticmethod
    def ode(t, x, u):
        u1 = u[0]
        x1, x2, x3, x4 = x

        # parameters
        b = 0.2  # m
        c = 0.3  # m
        r = 0.05  # m
        m_w = 0.5  # kg
        m_p = 3.0  # kg
        J_w = 0.5 * m_w * r ** 2  # kg*m2
        J_p = 1 / 12 * m_p * (b ** 2 + c ** 2)  # kg*m2
        g = 9.81  # m/s2

        q_dot = np.array([x3, x4])

        # motion dynamics in matrix form
        M = np.array([[2 * m_w + 2 * J_w / (r ** 2) + m_p, m_p * c * np.cos(x2)],
                      [m_p * c * np.cos(x2), J_p + m_p * c ** 2]])
        C = np.array([[0, -m_p * c * np.sin(x2) * x4], [0, 0]])
        G = np.array([0, -m_p * c * g * np.sin(x2)])
        F = np.array([2 / r * u1, 0])

        dqdt = np.dot(np.linalg.inv(M), (F - np.dot(C, q_dot) - G))
        dxdt = [x3, x4, dqdt[0], dqdt[1]]
        return dxdt

    def animation(self, episode, meanCost):

        def init():
            line.set_data([], [])
            #torque.set_data([], [])
            wheel.center = (0, 0)
            ax.add_patch(wheel)
            time_text.set_text('')
            return line, time_text, wheel#, torque

        # line and text
        def animate(t):
            # animation function
            thisx = [x_cart[t], x_tip[t] + x_cart[t]]
            thisy = [0.08, y_tip[t] + 0.08]
            #line.set_color('k')
            #wheel.set_color('k')
            line.set_data(thisx, thisy)
            #torque.set_data([self.history[t] / max(abs(self.history)), 0], [-0.05, -0.05])
            wheel.center = (thisx[0], 0.08)
            time_text.set_text(time_template % self.tt[t])
            return line, time_text, wheel#,torque

        # mapping from theta and s to the x,y-plane
        def cart_pole_plot(l, x):
            x_tip = l * np.sin(x[:, 1])
            x_cart = x[:, 0]
            y_tip = l * np.cos(x[:, 1])
            return x_tip, y_tip, x_cart

        # animation
        [x_tip, y_tip, x_cart] = cart_pole_plot(.5, self.history)
        fig, (ax) = plt.subplots(1, 1)
        ax.set_aspect('equal')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1, 1])
        plt.title('Episode ' + str(episode) + ', mean cost: ' + str(meanCost))
        rail, = ax.plot([-1.1, 1.1], [0, 0], 'ks-')
        torque, = ax.plot([], [], '-', color='r', lw=4)
        line, = ax.plot([], [], 'o-', color='k')
        wheel = plt.Circle((0, 1), 0.08, color='k', fill=False, lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)),
                                      interval=self.tt[1] * 1000, blit=False, init_func=init)
        return ani

#class AcroBot(StateSpaceModel):
#class Building(StateSpaceModel):
#class Ball(StateSpaceModel):
