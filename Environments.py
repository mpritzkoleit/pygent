from abc import abstractmethod
from numba import jit
import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.patches as patches
from scipy.integrate import solve_ivp
from Models.CartPoleDoubleParallel import modeling as cartPoleDoubleParallelODE
from Models.CartPole import modeling as cartPoleODE
from Utilities import  observation
import torch

class Environment(object):
    """ Base class for an environment.

    Attributes:

        x (array): current state x[k] (size = n)
        x_ (array): previous state x[k-1](size = n)
        history (array): previous states (x[0],x[1],...,x[k-1])
        tt (list): time vector (corresponding to history)
        terminated (bool): True, if environment is in a terminal state

    """

    def __init__(self, x0, uDim):
        if callable(x0):
            self.x0 = x0  # initial state
            x0 = x0()
        else:
            x0 = list(x0)
            self.x0 = x0
        self.x = x0  # current state
        self.x_ = x0 # previous state x[k-1]
        self.xDim = len(x0) # state dimension
        self.oDim = self.xDim # observation dimension
        self.uDim = uDim # inputs
        self.history = np.array([x0])
        self.tt = [0]
        self.terminated = False
        self.uMax = np.ones(uDim)

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

    def plot(self):
        """ Plots the environments history

        Returns:
            fig (matplotlib.pyplot.figure)
            ax (matploltib.pyplot.axes)

        """

        fig, ax = plt.subplots(len(self.x), 1, sharex='col')
        # Plot state trajectories
        fig.suptitle('States')
        if len(self.x) > 1:
            for i in range(len(self.x)):
                ax[i].step(self.tt, self.history[:, i], label=r'$x_'+str(i+1)+'$')
                ax[i].grid(True)
                ax[i].legend(loc='upper right')
        else:
            ax.step(self.tt, self.history[:, i], label=r'$x_1$')
            ax.grid(True)
            ax.legend(loc='upper right')
        plt.xlabel('t in s')
        # Todo: save data in numpy arrays
        return fig, ax

    def animation(self, episode, meanCost):
        pass

class OpenAIGym(Environment):
    """Environment subclass that uses a state space model of the form dx/dt = f(x,u)

    Attributes:
        ode (function): ODE for simulation
        cost (function): ODE for simulation

    """

    def __init__(self, id, render=True):
        self.env = gym.make(id)
        x0 = self.env.reset()
        uDim = self.env.action_space.shape[0]
        super(OpenAIGym, self).__init__(list(x0), uDim)
        self.o_ = self.x_
        self.o = self.x
        self.render = render

    def step(self, dt, u):
        """ Simulates the environment for 1 step of time t.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (array): control/action

        Returns:
            c (float): cost of state transition

        """
        if self.render:
            self.env.render()

        self.x_ = self.x  # shift state (x[k-1] = x[k])
        self.o_ = self.o
        # system simulation
        x, r, terminate, info = self.env.step(u)
        c = -r
        self.x = list(x)
        self.o = self.x
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        self.terminated = terminate
        return c

    def reset(self, x0):
        x0 = list(self.env.reset())
        self.history = np.array([x0])
        self.x_ = x0
        self.x = x0
        self.tt = [0]
        self.terminated = False


class StateSpaceModel(Environment):
    """Environment subclass that uses a state space model of the form dx/dt = f(x,u)

    Attributes:
        ode (function): ODE for simulation
        cost (function): ODE for simulation

    """

    def __init__(self, ode, cost, x0, uDim):
        super(StateSpaceModel, self).__init__(x0, uDim)
        self.ode = ode
        self.cost = cost
        self.xIsAngle = np.zeros([len(self.x_)], dtype=bool)
        self.o = self.x
        self.o_ = self.x_
        self.oDim = len(self.o)  # observation dimensions

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
        self.o_ = self.o
        y = list(sol.y[:, -1])  # extract simulation result
        self.x = self.mapAngles(y)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        c = self.cost(self.x_, u, self.x)*dt
        self.terminated = self.terminate(self.x_)
        self.o = observation(self.x, self.xIsAngle)
        return c

    def terminate(self, x):
        return False


    def fast_step(self, dt, u):
        """ Simulates the environment for 1 step of time t.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (array): control/action

        Returns:
            c (float): cost of state transition

        """

        self.x_ = self.x  # shift state (x[k-1] = x[k])
        self.o_ = self.o
        # euler step
        y = self.x_ + dt*self.ode(None, self.x_, u)
        self.x = self.mapAngles(y)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        c = self.cost(self.x_, u, self.x)*dt
        self.terminated = self.terminate(self.x_)
        self.o = observation(self.x, self.xIsAngle)
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


class Pendulum(StateSpaceModel):

    def __init__(self, cost, x0):
        super(Pendulum, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [True, False]
        self.o = observation(self.x, self.xIsAngle)
        self.o_ = self.o
        self.oDim = len(self.o)  # observation dimensions
        self.uMax = 5*np.ones(1)

    @staticmethod
    def ode(t, x, u):

        g = 9.81  # gravity
        b = 0.1  # dissipation
        u1, = u  # torque
        x1, x2 = x

        dx1dt = x2
        dx2dt = u1 + g * np.sin(x1) - b * x2

        return np.array([dx1dt, dx2dt])

    def terminate(self, x):
        x1, x2 = x
        if abs(x2) > 25:
            return True
        else:
            return False


    def animation(self, episode, meanCost):
        # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
        def pendulum_plot(l, xt):
            x_pole_end = -l * np.sin(xt[:, 0])
            y_pole_end = l * np.cos(xt[:, 0])

            return x_pole_end, y_pole_end

        # line and text
        def animate(t):
            thisx = [0, x_pole_end[t]]
            thisy = [0, y_pole_end[t]]

            pole.set_data(thisx, thisy)
            time_text.set_text(time_template % self.tt[t])
            return pole, time_text,

        x_pole_end, y_pole_end  = pendulum_plot(0.5, self.history)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.ylim((-.6, .6))
        plt.xlim((-.6, .6))
        plt.title('Pendulum')
        plt.xticks([], [])
        plt.yticks([], [])
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 1.05, '', transform=ax.transAxes)
        pole, = ax.plot([], [], 'b-', zorder=1, lw=3)
        circ = patches.Circle((0, 0), 0.03, fc='b', zorder=1)
        ax.add_artist(circ)
        # animation using matplotlibs animation library
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                      blit=True)
        return ani

class CartPole(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = cartPoleODE()
        super(CartPole, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o)  # observation dimensions
        self.o_ = self.o
        self.uMax = 10*np.ones(1)


    def terminate(self, x):
        x1, x2, x3, x4 = x
        if abs(x1) > 1:
            return True
        else:
            return False


    def animation(self, episode, meanCost):
        # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
        def cart_pole_plot(l, xt):
            x_pole_end = -l * np.sin(xt[:, 1]) + xt[:, 0]
            y_pole_end = l * np.cos(xt[:, 1])
            x_cart = xt[:, 0]

            return x_pole_end, y_pole_end, x_cart

        # line and text
        def animate(t):
            thisx = [x_cart[t], x_pole_end[t]]
            thisy = [0, y_pole_end[t]]

            pole.set_data(thisx, thisy)
            cart.set_xy([x_cart[t] - 0.1, -0.05])
            time_text.set_text(time_template % self.tt[t])
            return pole, cart, time_text,

        x_pole_end, y_pole_end, x_cart = cart_pole_plot(0.5, self.history)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set(xlabel=r'$x_1$')
        plt.ylim((-.6, .6))
        plt.yticks([], [])
        plt.title('CartPole')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 1.05, '', transform=ax.transAxes)
        rail, = ax.plot([min(-1, 1.2 * min(x_cart)), max(1, 1.2 * max(x_cart))], [0, 0], 'ks-', zorder=0)
        pole, = ax.plot([], [], 'b-', zorder=1, lw=3)
        cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
        ax.add_artist(cart)
        # animation using matplotlibs animation library
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                      blit=True)
        return ani


class CartPoleDoubleParallel(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = cartPoleDoubleParallelODE()
        super(CartPoleDoubleParallel, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, True, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        self.uMax = 15*np.ones(1)

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6 = x
        if abs(x1) > 1 or abs(x5) > 25 or abs(x6) > 25:
            return True
        else:
            return False

    def animation(self, episode, meanCost):
            # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
            def cart_pole_plot(l1, l2, xt):
                x_pole1_end = -l1 * np.sin(xt[:, 1]) + xt[:, 0]
                y_pole1_end = l1 * np.cos(xt[:, 1])
                x_pole2_end = -l2 * np.sin(xt[:, 2]) + xt[:, 0]
                y_pole2_end = l2 * np.cos(xt[:, 2])
                x_cart = xt[:, 0]

                return x_pole1_end, y_pole1_end, x_pole2_end, y_pole2_end, x_cart

            # line and text
            def animate(t):
                thisx1 = [x_cart[t], x_pole1_end[t]]
                thisy1 = [0, y_pole1_end[t]]
                thisx2 = [x_cart[t], x_pole2_end[t]]
                thisy2 = [0, y_pole2_end[t]]

                pole1.set_data(thisx1, thisy1)
                pole2.set_data(thisx2, thisy2)
                cart.set_xy([x_cart[t] - 0.1, -0.05])
                time_text.set_text(time_template % self.tt[t])
                return pole1, pole2, cart, time_text,

            x_pole1_end, y_pole1_end, x_pole2_end, y_pole2_end, x_cart = cart_pole_plot(0.5, .7, self.history)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set(xlabel=r'$x_1$')
            plt.ylim((-.8, .8))
            plt.yticks([], [])
            plt.title('CartPoleDoubleParallel')
            time_template = 'time = %.1fs'
            time_text = ax.text(0.05, 1.05, '', transform=ax.transAxes)
            rail, = ax.plot([min(-1, 1.2 * min(x_cart)), max(1, 1.2 * max(x_cart))], [0, 0], 'ks-', zorder=0)
            pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole2, = ax.plot([], [], 'b-', zorder=1, lw=3)
            cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
            ax.add_artist(cart)
            # animation using matplotlibs animation library
            ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                          blit=True)
            return ani

class Car(StateSpaceModel):

    def __init__(self, cost, x0):
        super(Car, self).__init__(self.ode, cost, x0, uDim=2)
        self.xIsAngle = [False, False, True, False]
        self.o = observation(self.x, self.xIsAngle)
        self.o_ = self.o
        self.oDim = len(self.o)  # observation dimensions
        self.uMax = np.array([.5, 2.])

    @staticmethod
    def ode(t, x, u):

        g = 9.81  # gravity
        d = 2.0  # dissipation
        u1, u2 = u  # torque
        x1, x2, x3 = x

        dx1dt = np.cos(x3)*u1
        dx2dt = np.sin(x3)*u1
        dx3dt = np.tan(u2)*u1/d

        return np.array([dx1dt, dx2dt, dx3dt])

    def terminate(self, x):
            return False
#class AcroBot(StateSpaceModel):
#class Building(StateSpaceModel):
#class Ball(StateSpaceModel):
# #class DoubleCartPole
#class TripleCartPole
