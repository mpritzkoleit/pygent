import numpy as np
np.random.seed(0)
from abc import abstractmethod, abstractproperty
import matplotlib.pyplot as plt

class Agent(object):
    """ Base class for an agent. 
    
    Attributes:
        uDim (int): dimension of control input
    """

    def __init__(self, uDim):
        self.tt = [0] # array with time stamps
        self.history = np.zeros([1, uDim]) # agents history (the control trajectory)
        self.uDim = uDim # dimension of control input
        # todo: add dt as argument

    @abstractmethod
    def take_action(self, *args):
        """ Abstract method for taking an action / applying a control. """

        return

    def control(self, dt, u):
        """ Apply the given control/action.

                Args:
                    dt (int, float): duration of step (not solver step size)
                    u (array): control/action

                Returns:
                    u (array): control/action

                """
        # todo: dt is optional
        self.u = u
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u

    def reset(self):
        """ Reset the agents history. """

        self.tt = [0]
        self.history = np.zeros([1, self.uDim])
        pass

    def plot(self):
        """ Plots the agents history (the control trajectory)

        Returns:
            fig (matplotlib.pyplot.figure)
            ax (matplotlib.pyplot.axes)

        """

        fig, ax = plt.subplots(self.uDim, 1, dpi=150)
        # plot control trajectories
        if self.uDim > 1:
            for i, axes in enumerate(ax):
                ax[i].step(self.tt, self.history[:, i], 'b', lw=1, sharex=True)
                ax[i].grid(True)
                ax[i].set_ylabel(r'$u_'+str(i+1)+'$')
        else:
            # single input case
            ax.step(self.tt, self.history, 'b', lw=1)
            ax.set_ylabel(r'$u_1$')
            ax.grid(True)
        plt.xlabel(r't[s]')
        plt.tight_layout()
        # Todo: save data in numpy arrays
        return fig, ax

class FeedBack(Agent):
    """Agent subclass: a standard state feedback of the form u = mu(x)

        Attributes:
            mu (function): feedback law
        """

    def __init__(self, mu, uDim):
        super(FeedBack, self).__init__(uDim)
        self.mu = mu

    def take_action(self, dt, x):
        """ Computes control/action defined by the feedback law mu(x).
        Adds the control/action to the agents history.

        Args:
            dt (int, float): duration of step (not solver step size)
            x (array): state of the system/environment

        Returns:
            u (array): control/action

        """

        # todo: dt is optional
        self.u = self.mu(x)  # compute control/action signal
        self.history = np.concatenate((self.history, np.array([self.u])))  # save current action in history
        self.tt.extend([self.tt[-1] + dt])  # increment simulation time
        return self.u



