__name__ == "pygent.environments"

from abc import abstractmethod
import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.integrate import solve_ivp
import inspect
import pickle
import copy

# from PyGent
from pygent.modeling_scripts.cart_pole_double_parallel import load_existing as cart_pole_double_parallel_ode
from pygent.modeling_scripts.cart_pole_double_serial import load_existing as cart_pole_double_serial_ode
from pygent.modeling_scripts.cart_pole_triple import load_existing as cart_pole_triple_ode
from pygent.modeling_scripts.cart_pole import load_existing as cart_pole_ode
from pygent.modeling_scripts.acrobot import load_existing as acrobot_ode
from pygent.helpers import observation, mapAngles

class Environment(object):
    """ Environment base class.

    Args:
        x0 (array, list, callable):

    Attributes:

        x (array): current state x[k] (size = n)
        x_ (array): previous state x[k-1](size = n)
        history (array): previous states (x[0],x[1],...,x[k-1])
        tt (list): time vector (corresponding to history)
        terminated (bool): True, if environment is in a terminal state

    """

    def __init__(self, x0, uDim, dt):
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
        self.xIsAngle = np.zeros([self.xDim], dtype=bool)
        self.history = np.array([x0])
        self.tt = [0]
        self.terminated = False
        self.uMax = np.ones(uDim)
        self.dt = dt

    def get_state(self):
        return self.x

    def reset(self, x0):
        """ Resets environment to state x0

        Args:
            x0 (array, list, callable): initial state

        """
        if callable(x0):
            x0 = x0()
        self.history = np.array([x0])
        self.x_ = x0
        self.x = x0
        self.tt = [0]
        self.terminated = False
        pass

    @abstractmethod
    def step(self, *args):
        return

    def plot(self):
        """ Plots the environments history

        Returns:
            fig (matplotlib.pyplot.figure)
            ax (matploltib.pyplot.axes)

        """

        fig, ax = plt.subplots(len(self.x), 1, dpi=300, sharex=True)
        # Plot state trajectories
        if len(self.x) > 1:
            for i in range(len(self.x)):
                ax[i].step(self.tt, self.history[:, i], 'b',  lw=1)
                ax[i].set_ylabel(r'$x_'+str(i+1)+'$')
                ax[i].grid(True)
                if self.xIsAngle[i]:
                    ax[i].yaxis.set_major_formatter(FuncFormatter(
                        lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
                    ax[i].yaxis.set_major_locator(MultipleLocator(base=np.pi))
        else:
            ax.step(self.tt, self.history[:, 0], 'b',  lw=1)
            ax.grid(True)
            plt.ylabel(r'$x_1$')
        fig.align_ylabels(ax)
        plt.xlabel(r't[s]')
        plt.tight_layout()
        # Todo: save data in numpy arrays
        return fig, ax

    def save_history(self, filename, path):
        history_dict = {'tt': self.tt, 'xx': self.history}
        pickle.dump(history_dict, open(path + filename +'.p', 'wb'))
        pass

    def animation(self):
        pass

    def observe(self, x):
        return x

class OpenAIGym(Environment):
    """ Environment subclass, that is a wrapper for an 'OpenAI gym' environment.

    Attributes:
        ode (function): ODE for simulation
        cost (function): ODE for simulation

    """

    def __init__(self, id, render=True):
        self.env = gym.make(id)
        x0 = self.env.reset()
        uDim = self.env.action_space.shape[0]
        super(OpenAIGym, self).__init__(list(x0), uDim, self.env.dt)
        self.uMax = self.env.action_space.high[0]*np.ones(uDim)
        self.o_ = self.x_
        self.o = self.x
        self.render = render

    def step(self, *args):
        """ Simulates the environment for 1 step of time t.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (list, ndarray): control/action

        Returns:
            c (float): cost of state transition

        """

        if args.__len__()==2:
            u = args[0]
            dt = args[1]
        elif args.__len__() == 1:
            u = args[0]
            dt = self.dt

        if self.render:
            self.env.render()

        self.x_ = self.x  # shift state (x[k-1] = x[k])
        self.o_ = self.o
        x, r, terminate, info = self.env.step(u)
        c = -r # cost = - reward
        self.x = list(x)
        self.o = self.x
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        self.terminated = terminate
        return c*dt

    def reset(self, x0):
        x0 = list(self.env.reset())
        self.history = np.array([x0])
        self.x_ = x0
        self.x = x0
        self.tt = [0]
        self.terminated = False

class StateSpaceModel(Environment):
    """ Environment subclass that uses a state space model of the form dx/dt = f(x, u)
    to represent the environments dynamics.

    Args:
        ode
        cost
        x0
        uDim

    Attributes:
        ode (function): ODE for simulation
        cost (function): cost function (returns scalar)
        xIsAngle (ndarray): 'True' if state is an angle, 'False' otherwise
        o
        o_
        oDim

    """

    def __init__(self, ode, cost, x0, uDim, dt,
                 terminal_cost=0.):
        super(StateSpaceModel, self).__init__(x0, uDim, dt)
        self.ode = ode
        params = inspect.signature(cost).parameters
        cost_args = params.__len__()
        if cost_args == 1:
            self.cost = lambda x_, u_, x, t, mod: cost(x_)
        elif cost_args == 2:
            if 'mod' in params:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, mod)
            elif 't' in params:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, t)
            else:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_)
        elif cost_args == 3:
            if 'mod' in params:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_, mod)
            elif 't' in params:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_, t)
            else:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_, x)
        elif cost_args == 4:
            if 'mod' in params and 't' in params:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_, t, mod)
            elif 'mod' in params and not 't' in params:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_, x, mod)
            else:
                self.cost = lambda x_, u_, x, t, mod: cost(x_, u_, x, t)
        elif cost_args == 5:
            self.cost = cost
        else:
            print('Cost function must to be of the form c(x_, u_, x, t, mod), where mod is numpy/sympy.')
            assert(True)
        self.xIsAngle = np.zeros([len(self.x_)], dtype=bool)
        self.o = self.x
        self.o_ = self.x_
        self.oDim = len(self.o)  # observation dimensions
        self.terminal_cost = terminal_cost

    def step(self, *args):
        """ Simulates the environment for 1 step of time t.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (array): control/action

        Returns:
            c (float): cost of state transition

        """
        self.x_ = self.x  # shift state (x[k-1] = x[k])
        self.o_ = self.o
        if args.__len__()==2:
            u = args[0]
            dt = args[1]
        elif args.__len__() == 1:
            u = args[0]
            dt = self.dt

        # system simulation
        sol = solve_ivp(lambda t, x: self.ode(t, x, u), (0, dt), self.x_, 'RK45')
        # todo: only output value of the last timestep
        y = list(sol.y[:, -1])  # extract simulation result
        self.x = y
        self.o = self.observe(self.x)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        self.terminated = self.terminate(self.x)
        #x_2pi = mapAngles(self.xIsAngle, self.x_)
        #x2pi = mapAngles(self.xIsAngle, self.x)
        #c = (self.cost(x_2pi, u, x2pi, np) + self.terminal_cost*self.terminated)*dt
        t = self.tt[-1]
        c = (self.cost(self.x_, u, self.x, t, np) + self.terminal_cost * self.terminated) * dt
        return c

    def terminate(self, x):
        """ Check if a 'terminal' state is reached.

            Args:
                x (ndarray, list): state

            Returns:
                terminated (bool): 'True' if 'x' is a terminal state. """

        terminated = False
        return terminated


    def fast_step(self, *args):
        """ Simulates the environment for 1 step of time t, using Euler forward integration.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (array): control/action

        Returns:
            c (float): cost of state transition

        """

        if args.__len__()==2:
            u = args[0]
            dt = args[1]
        elif args.__len__() == 1:
            u = args[0]
            dt = self.dt

        self.x_ = self.x  # shift state (x[k-1] := x[k])
        self.o_ = self.o

        # Euler forward step
        y = self.x_ + dt*self.ode(None, self.x_, u)
        self.x = y
        self.o = self.observe(self.x)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        self.terminated = self.terminate(self.x)
        t = self.tt[-1]
        c = (self.cost(self.x_, u, self.x, t, np) + self.terminal_cost*self.terminated)*dt
        return c

    def observe(self, x):
        obsv = observation(x, self.xIsAngle)
        return obsv

class Pendulum(StateSpaceModel):

    def __init__(self, cost, x0, dt):
        super(Pendulum, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [True, False]
        self.o = self.observe(self.x)
        self.o_ = self.o
        self.oDim = len(self.o)  # observation dimensions
        self.uMax = 3.5*np.ones(1)

    @staticmethod
    def ode(t, x, u):

        g = 9.81  # gravity
        b = 0.02  # dissipation
        u1, = u  # torque
        x1, x2 = x

        dx1dt = x2
        dx2dt = u1 + g*np.sin(x1) - b*x2

        return np.array([dx1dt, dx2dt])

    def terminate(self, x):
        x1, x2 = x
        if abs(x2) > 10 or abs(x1)>8*np.pi:
            return True
        else:
            return False


    def animation(self):
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

class AngularPendulum(StateSpaceModel):

    def __init__(self, cost, x0, dt):
        super(AngularPendulum, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [False, False, False]
        self.o = self.observe(self.x)
        self.o_ = self.o
        self.oDim = len(self.o)  # observation dimensions
        self.uMax = 3.5*np.ones(1)

    @staticmethod
    def ode(t, x, u):

        g = 9.81  # gravity
        b = 0.02  # dissipation
        u1, = u  # torque
        x1, x2, x3 = x

        dx1dt = -x2*x3
        dx2dt = x1*x3
        dx3dt = u1 + g*x2 - b*x3

        return np.array([dx1dt, dx2dt, dx3dt])

    def terminate(self, x):
        x1, x2, x3 = x
        if abs(x3) > 10:
            return True
        else:
            return False


    def animation(self):
        # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
        def pendulum_plot(l, xt):
            x_pole_end = -l * xt[:, 1]
            y_pole_end = l * xt[:, 0]

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
    def __init__(self, cost, x0, dt, linearized=True):
        self.ode, self.A, self.B = cart_pole_ode(linearized=linearized)
        super(CartPole, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [False, True, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o)  # observation dimensions
        self.o_ = self.o
        if linearized:
            self.uMax = 10*np.ones(1) # max. acceleration
        else:
            self.uMax = 100*np.ones(1) # max. force
        self.x1Max = 1.2
        self.x3Max = 4.0


    def terminate(self, x):
        x1, x2, x3, x4 = x
        if np.abs(x1) > self.x1Max or np.abs(x3)> self.x3Max:
            return True
        else:
            return False


    def animation(self):
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
        fig, ax = plt.subplots(dpi=150)
        ax.set_aspect('equal')
        ax.set(xlabel=r'$x_1$')
        plt.ylim((-.6, .6))
        #plt.xlim((-self.x1Max-0.2, self.x1Max+0.2))
        plt.yticks([], [])
        plt.title('CartPole')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 1.05, '', transform=ax.transAxes)
        rail, = ax.plot([-self.x1Max-0.1, self.x1Max+0.1], [0, 0], 'ks-', zorder=0)
        pole, = ax.plot([], [], 'b-', zorder=1, lw=3)
        cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
        ax.add_artist(cart)
        # animation using matplotlibs animation library
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                      blit=True)
        return ani

class CartPoleDoubleSerial(StateSpaceModel):
    def __init__(self, cost, x0, dt, linearized=True, task='swing_up'):
        self.ode, self.A, self.B = cart_pole_double_serial_ode(linearized=linearized)
        super(CartPoleDoubleSerial, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [False, True, True, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        if linearized:
            self.uMax = 15*np.ones(1)
        else:
            self.uMax = 100*np.ones(1)
        self.task = task
        self.x1Max = 1.5

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6 = x
        if self.task == 'swing_up':
            if abs(x1) > self.x1Max or abs(x5) > 25 or abs(x6) > 25:
                return True
            else:
                return False
        elif self.task == 'balance':
            if abs(x1) > self.x1Max or abs(x2) > 1. or abs(x3) > 1. or abs(x5) > 25 or abs(x6) > 25:
                return True
            else: 
                return False
        else:
            return False

    def animation(self):
            # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
            def cart_pole_plot(l1, l2, xt):
                x_pole1_end = -l1 * np.sin(xt[:, 1]) + xt[:, 0]
                y_pole1_end = l1 * np.cos(xt[:, 1])
                x_pole2_end = x_pole1_end -l2 * np.sin(xt[:, 2])
                y_pole2_end = y_pole1_end +l2 * np.cos(xt[:, 2])
                x_cart = xt[:, 0]

                return x_pole1_end, y_pole1_end, x_pole2_end, y_pole2_end, x_cart

            # line and text
            def animate(t):
                thisx1 = [x_cart[t], x_pole1_end[t]]
                thisy1 = [0, y_pole1_end[t]]
                thisx2 = [x_pole1_end[t], x_pole2_end[t]]
                thisy2 = [y_pole1_end[t], y_pole2_end[t]]

                pole1.set_data(thisx1, thisy1)
                pole2.set_data(thisx2, thisy2)
                cart.set_xy([x_cart[t] - 0.1, -0.05])
                time_text.set_text(time_template % self.tt[t])
                return pole1, pole2, cart, time_text,

            x_pole1_end, y_pole1_end, x_pole2_end, y_pole2_end, x_cart = cart_pole_plot(0.323, .419, self.history)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set(xlabel=r'$x_1$')
            plt.ylim((-1.1, 1.1))
            plt.yticks([], [])
            plt.title('CartPoleDoubleSerial')
            time_template = 'time = %.1fs'
            time_text = ax.text(0.0, 1.05, '', transform=ax.transAxes)
            rail, = ax.plot([-(self.x1Max+0.2), self.x1Max+0.2], [0, 0], 'ks-', zorder=0)
            pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole2, = ax.plot([], [], 'b-', zorder=1, lw=3)
            cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
            ax.add_artist(cart)
            # animation using matplotlibs animation library
            ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                          blit=True)
            return ani

class CartPoleDoubleParallel(StateSpaceModel):
    def __init__(self, cost, x0, dt, linearized=True):
        self.ode, self.A, self.B = cart_pole_double_parallel_ode(linearized=linearized)
        super(CartPoleDoubleParallel, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [False, True, True, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        if linearized:
            self.uMax = 25 * np.ones(1)
        else:
            self.uMax = 100 * np.ones(1)
        self.x1Max = 1.5

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6 = x
        if abs(x1) > self.x1Max or abs(x5) > 25 or abs(x6) > 25:
            return True
        else:
            return False

    def animation(self):
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
            rail, = ax.plot([-(self.x1Max+0.1), self.x1Max+0.1], [0, 0], 'ks-', zorder=0)
            pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole2, = ax.plot([], [], 'b-', zorder=1, lw=3)
            cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
            ax.add_artist(cart)
            # animation using matplotlibs animation library
            ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                          blit=True)
            return ani

class CartPoleTriple(StateSpaceModel):
    def __init__(self, cost, x0, dt, linearized=True):
        self.ode, self.A, self.B  = cart_pole_triple_ode(linearized=linearized)
        super(CartPoleTriple, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [False, True, True, True, False, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        if linearized:
            self.uMax = 20 * np.ones(1)
        else:
            self.uMax = 100 * np.ones(1)
        self.x1Max = 1.5

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        if abs(x1) > self.x1Max:
            return True
        else:
            return False

    def animation(self):
            # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
            def cart_pole_plot(l1, l2, l3, xt):
                x_p1 = -l1*np.sin(xt[:, 1])+xt[:, 0]
                y_p1 = l1*np.cos(xt[:, 1])
                x_p2 = x_p1 - l2*np.sin(xt[:, 2])
                y_p2 = y_p1 + l2*np.cos(xt[:, 2])
                x_p3 = x_p2 - l3*np.sin(xt[:, 3])
                y_p3 = y_p2 + l3*np.cos(xt[:, 3])
                x_cart = xt[:, 0]

                return x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_cart

            # line and text
            def animate(t):
                thisx1 = [x_cart[t], x_p1[t]]
                thisy1 = [0, y_p1[t]]
                thisx2 = [x_p1[t], x_p2[t]]
                thisy2 = [y_p1[t], y_p2[t]]
                thisx3 = [x_p2[t], x_p3[t]]
                thisy3 = [y_p2[t], y_p3[t]]

                pole1.set_data(thisx1, thisy1)
                pole2.set_data(thisx2, thisy2)
                pole3.set_data(thisx3, thisy3)
                cart.set_xy([x_cart[t] - 0.1, -0.05])
                time_text.set_text(time_template % self.tt[t])
                return pole1, pole2, pole3, cart, time_text,

            x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_cart = cart_pole_plot(0.323, 0.419, 0.484, self.history)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set(xlabel=r'$x_1$')
            plt.ylim((-1.4, 1.4))
            plt.yticks([], [])
            plt.title('CartPoleTriple')
            time_template = 'time = %.1fs'
            time_text = ax.text(0.0, 1.05, '', transform=ax.transAxes)
            rail, = ax.plot([-(self.x1Max+0.1), self.x1Max+0.1], [0, 0], 'ks-', zorder=0)
            pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole2, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole3, = ax.plot([], [], 'b-', zorder=1, lw=3)
            cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
            ax.add_artist(cart)
            # animation using matplotlibs animation library
            ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                          blit=True)
            return ani

class Car(StateSpaceModel):
    # Todo: from iLQG paper
    def __init__(self, cost, x0, dt):
        super(Car, self).__init__(self.ode, cost, x0, 2, dt)
        self.xIsAngle = [False, False, True, False]
        self.o = observation(self.x, self.xIsAngle)
        self.o_ = self.o
        self.oDim = len(self.o)  # observation dimensions
        self.uMax = np.array([.5, 2.])

    @staticmethod
    def ode(t, x, u):

        d = 2.0  # dissipation
        h = 0.03
        u1, u2 = u  # a, w
        x1, x2, x3, x4 = x # x, y, theta, v
        f = h*x4
        b = f*np.cos(u1) + d - np.sqrt(d**2 - f**2*np.sin(u1)**2)
        dx1dt = 1/h*np.cos(x3)*b
        dx2dt = 1/h*np.sin(x3)*b
        dx3dt = 1/h*np.arcsin(np.sin(u1)*f/d)
        dx4dt = u2

        return np.array([dx1dt, dx2dt, dx3dt, dx4dt])

    def terminate(self, x):
        x1, x2, x3, x4 = x
        if abs(x1) > 5 or abs(x2) > 5:
            return True
        else:
            return False

class Acrobot(StateSpaceModel):
    def __init__(self, cost, x0, dt, linearized=True):
        self.ode, self.A, self.B = acrobot_ode(linearized=linearized)
        super(Acrobot, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [True, True, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o)  # observation dimensions
        self.o_ = self.o
        if linearized:
            self.uMax = 50*np.ones(1)
        else:
            self.uMax = 5*np.ones(1)


    def terminate(self, x):
        x1, x2, x3, x4 = x
        if abs(x3) > 50 or abs(x4) > 50:
            return True
        else:
            return False


    def animation(self):
        # mapping from theta and s to the x,y-plane (definition of the line points, that represent the pole)
        def acrobot_plot(l0, l1, xt):
            x_pole0 = -l0*np.sin(xt[:, 0])
            y_pole0 = l0*np.cos(xt[:, 0])
            x_pole1 = x_pole0 -l1*np.sin(xt[:, 0]+xt[:, 1])
            y_pole1 = y_pole0 + l1*np.cos(xt[:, 0]+xt[:, 1])

            return x_pole0, y_pole0, x_pole1, y_pole1,

        # line and text
        def animate(t):
            thisx0 = [0, x_pole0[t]]
            thisy0 = [0, y_pole0[t]]
            thisx1 = [x_pole0[t], x_pole1[t]]
            thisy1 = [y_pole0[t], y_pole1[t]]

            pole0.set_data(thisx0, thisy0)
            pole1.set_data(thisx1, thisy1)
            time_text.set_text(time_template % self.tt[t])
            return pole0, pole1, time_text,

        x_pole0, y_pole0, x_pole1, y_pole1 = acrobot_plot(0.5, 0.5, self.history)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.ylim((-1.1, 1.1))
        plt.xlim((-1.1, 1.1))
        plt.yticks([], [])
        plt.xticks([], [])
        plt.title('Acrobot')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 1.05, '', transform=ax.transAxes)
        pole0, = ax.plot([], [], 'b-o', zorder=1, lw=3)
        pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
        # animation using matplotlibs animation library
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                      blit=True)
        return ani

#class Building(StateSpaceModel):

class MarBot(StateSpaceModel):
    def __init__(self, cost, x0, dt):
        super(MarBot, self).__init__(self.ode, cost, x0, 1, dt)
        self.xIsAngle = [False, True, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o)  # observation dimensions
        self.o_ = self.o
        self.uMax = 2*np.ones(1)

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

    def terminate(self, x):
        x1, x2, x3, x4 = x
        if np.abs(x1) > 1:
            return True
        else:
            return False

    def animation(self):
        def init():
            line.set_data([], [])
            # torque.set_data([], [])
            wheel.center = (0, 0)
            ax.add_patch(wheel)
            time_text.set_text('')
            return line, time_text, wheel  # , torque

        # line and text
        def animate(t):
            # animation function
            thisx = [x_cart[t], x_tip[t] + x_cart[t]]
            thisy = [0.08, y_tip[t] + 0.08]
            # line.set_color('k')
            # wheel.set_color('k')
            line.set_data(thisx, thisy)
            # torque.set_data([self.history[t] / max(abs(self.history)), 0], [-0.05, -0.05])
            wheel.center = (thisx[0], 0.08)
            time_text.set_text(time_template % self.tt[t])
            return line, time_text, wheel  # ,torque

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
        ax.set_xlim([-2.1, 2.1])
        ax.set_ylim([-0.2, 1])
        plt.yticks([], [])
        rail, = ax.plot([-1.7, 1.7], [0, 0], 'ks-')
        torque, = ax.plot([], [], '-', color='r', lw=4)
        line, = ax.plot([], [], 'o-', color='k')
        wheel = plt.Circle((0, 1), 0.08, color='k', fill=False, lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)),
                                      interval=self.tt[1] * 1000, blit=False, init_func=init)
        return ani