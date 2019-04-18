from abc import abstractmethod

class Algorithm(object):
    """ Algorithm base class.

    Attributes:
        environment (Environment(object)): environment
        agent (Agent(object)): agent of the algorithm
        t (int, float): episode length in seconds
        dt (int, float): step size in seconds
        steps (int): episode length int(t/dt)
        meanCost (list): mean cost per step of an episode
        totalCost (list): total cost of an episode
        episode (int): current episode

    """

    def __init__(self, environment, agent, t, dt):
        self.t = t
        self.dt = dt
        self.steps = int(t/dt)
        self.agent = agent
        self.environment = environment
        self.meanCost = []
        self.totalCost = []
        self.episode = 1

    @abstractmethod
    def run_episode(self):
        """ Abstract method. Run an episode/iteration"""
        return

    @abstractmethod
    def learning_curve(self):
        """ Abstract method. Display learning progress."""
        return