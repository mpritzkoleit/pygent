from abc import abstractmethod
import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.patches as patches
from scipy.integrate import solve_ivp
import inspect

# from PyGent
from modeling_scripts.cart_pole_double_parallel import load_existing as cart_pole_double_parallel_ode
from modeling_scripts.cart_pole_double_serial import load_existing as cart_pole_double_serial_ode
from modeling_scripts.cart_pole_triple import load_existing as cart_pole_triple_ode
from modeling_scripts.cart_pole import load_existing as cart_pole_ode
from modeling_scripts.acrobot import load_existing as acrobot_ode
from helpers import observation

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
        self.xIsAngle = np.zeros([self.xDim], dtype=bool)
        self.history = np.array([x0])
        self.tt = [0]
        self.terminated = False
        self.uMax = np.ones(uDim)

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

        fig, ax = plt.subplots(len(self.x), 1, sharex=True)
        # Plot state trajectories
        if len(self.x) > 1:
            for i in range(len(self.x)):
                ax[i].step(self.tt, self.history[:, i], 'b',  lw=1)
                ax[i].set_ylabel(r'$x_'+str(i+1)+'$')
                ax[i].grid(True)
        else:
            ax.step(self.tt, self.history[:, 0], 'b',  lw=1)
            ax.grid(True)
            plt.ylabel(r'$x_1$')
        plt.xlabel(r't[s]')
        plt.tight_layout()
        # Todo: save data in numpy arrays
        return fig, ax

    def animation(self, episode, meanCost):
        pass

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
        super(OpenAIGym, self).__init__(list(x0), uDim)
        self.o_ = self.x_
        self.o = self.x
        self.render = render

    def step(self, dt, u):
        """ Simulates the environment for 1 step of time t.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (list, ndarray): control/action

        Returns:
            c (float): cost of state transition

        """
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
        return c

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

    def __init__(self, ode, cost, x0, uDim):
        super(StateSpaceModel, self).__init__(x0, uDim)
        self.ode = ode
        cost_args = inspect.signature(cost).parameters.__len__()
        if cost_args == 1:
            self.cost = lambda x_, u_, x, mod: cost(x_)
        if cost_args == 2:
            self.cost = lambda x_, u_, x, mod: cost(x_, u_)
        elif cost_args == 3 and 'mod' in inspect.signature(cost).parameters:
            self.cost = lambda x_, u_, x, mod: cost(x_, u_, mod)
        elif cost_args == 3 and 'mod' not in inspect.signature(cost).parameters:
            self.cost = lambda x_, u_, x, mod: cost(x_, u_, x)
        elif cost_args == 4:
            self.cost = cost
        else:
            print('Cost function must to be of the form c(x), c(x, u), c(x_, u_, x), c(x, u, mod) or c(x_, u_, x, mod), where mod is a placeholder for numpy/sympy.')
            assert(True)
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
        self.x_ = self.x  # shift state (x[k-1] = x[k])
        self.o_ = self.o

        # system simulation
        sol = solve_ivp(lambda t, x: self.ode(t, x, u), (0, dt), self.x_)
        # todo: only output value of the last timestep
        y = list(sol.y[:, -1])  # extract simulation result
        self.x = self.mapAngles(y)
        self.o = self.observe(self.x)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        c = self.cost(self.x_, u, self.x, np)*dt
        self.terminated = self.terminate(self.x_)
        return c

    def terminate(self, x):
        """ Check if a 'terminal' state is reached.

            Args:
                x (ndarray, list): state

            Returns:
                terminated (bool): 'True' if 'x' is a terminal state. """

        terminated = False
        return terminated


    def fast_step(self, dt, u):
        """ Simulates the environment for 1 step of time t, using Euler forward integration.

        Args:
            dt (int, float): duration of step (not solver step size)
            u (array): control/action

        Returns:
            c (float): cost of state transition

        """

        self.x_ = self.x  # shift state (x[k-1] := x[k])
        self.o_ = self.o
        # Euler forward step
        y = self.x_ + dt*self.ode(None, self.x_, u)
        self.x = self.mapAngles(y)
        self.o = self.observe(self.x)
        self.history = np.concatenate((self.history, np.array([self.x])))  # save current state
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        c = self.cost(self.x_, u, self.x, np)*dt
        self.terminated = self.terminate(self.x_)
        return c

    def mapAngles(self, x):
        """ Maps angles to the interval [-pi,pi]. """

        for i in range(len(x)):
            if self.xIsAngle[i]:
                # map theta to [-pi,pi]
                if x[i] > np.pi:
                    x[i] -= 2*np.pi
                elif x[i] < -np.pi:
                    x[i] += 2*np.pi
        return x

    def observe(self, x):
        obsv = observation(x, self.xIsAngle)
        return obsv

class Pendulum(StateSpaceModel):

    def __init__(self, cost, x0):
        super(Pendulum, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [True, False]
        self.o = self.observe(self.x)
        self.o_ = self.o
        self.oDim = len(self.o)  # observation dimensions
        self.uMax = 2*np.ones(1)

    @staticmethod
    def ode(t, x, u):

        g = 9.81  # gravity
        b = 0.1  # dissipation
        u1, = u  # torque
        x1, x2 = x

        dx1dt = x2
        dx2dt = u1 + g*np.sin(x1) - b*x2

        return np.array([dx1dt, dx2dt])

    def terminate(self, x):
        x1, x2 = x
        if abs(x2) > 25:
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

class CartPole(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = cart_pole_ode()
        super(CartPole, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o)  # observation dimensions
        self.o_ = self.o
        self.uMax = 10*np.ones(1)


    def terminate(self, x):
        x1, x2, x3, x4 = x
        if np.abs(x1) > 1.0:
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
        plt.xlim((min(-1.4, 1.2 * min(x_cart)), max(1.4, 1.2 * max(x_cart))))
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

class CartPoleDoubleSerial(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = cart_pole_double_serial_ode()
        super(CartPoleDoubleSerial, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, True, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        self.uMax = 40*np.ones(1)

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6 = x
        if abs(x1) > 1 or abs(x5) > 25 or abs(x6) > 25:
            return True
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
            rail, = ax.plot([min(-1, 1.2 * min(x_cart)), max(1, 1.2 * max(x_cart))], [0, 0], 'ks-', zorder=0)
            pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole2, = ax.plot([], [], 'b-', zorder=1, lw=3)
            cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
            ax.add_artist(cart)
            # animation using matplotlibs animation library
            ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                          blit=True)
            return ani

class CartPoleDoubleParallel(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = cart_pole_double_parallel_ode()
        super(CartPoleDoubleParallel, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, True, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        self.uMax = 25*np.ones(1)

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6 = x
        if abs(x1) > 1 or abs(x5) > 25 or abs(x6) > 25:
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
            rail, = ax.plot([min(-1, 1.2 * min(x_cart)), max(1, 1.2 * max(x_cart))], [0, 0], 'ks-', zorder=0)
            pole1, = ax.plot([], [], 'b-', zorder=1, lw=3)
            pole2, = ax.plot([], [], 'b-', zorder=1, lw=3)
            cart = patches.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='b', zorder=1)
            ax.add_artist(cart)
            # animation using matplotlibs animation library
            ani = animation.FuncAnimation(fig, animate, np.arange(len(self.tt)), interval=self.tt[1] * 1000,
                                          blit=True)
            return ani

class CartPoleTriple(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = cart_pole_triple_ode()
        super(CartPoleTriple, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, True, True, False, False, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o) #observation dimensions
        self.o_ = self.o
        self.uMax = 20*np.ones(1)

    def terminate(self, x):
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        if abs(x1) > 0.7:
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
            rail, = ax.plot([min(-.8, 1.2 * min(x_cart)), max(.8, 1.2 * max(x_cart))], [0, 0], 'ks-', zorder=0)
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

    def __init__(self, cost, x0):
        super(Car, self).__init__(self.ode, cost, x0, uDim=2)
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
            return False

class Acrobot(StateSpaceModel):
    def __init__(self, cost, x0):
        self.ode = acrobot_ode()
        super(Acrobot, self).__init__(self.ode, cost, x0, uDim=1)
        self.xIsAngle = [False, True, False, False]
        self.o = observation(self.x, self.xIsAngle)
        self.oDim = len(self.o)  # observation dimensions
        self.o_ = self.o
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
