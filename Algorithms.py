from abc import abstractmethod

class Algorithm(object):
    """ Learning Process

    Attributes:
        t (int, float): episode length
        dt (int, float): step size
        meanCost (int, float): mean cost of an episode
        agent (Agent(object)): agent of the algorithm
        environment (Environment(object)): environment

    """

    def __init__(self, environment, agent, t, dt):
        self.t = t
        self.dt = dt
        self.steps = int(t/dt)
        self.agent = agent
        self.environment = environment
        self.meanCost = []
        self.episode = 1

    @abstractmethod
    def run_episode(self):
        return

    @abstractmethod
    def learning_curve(self):
        return