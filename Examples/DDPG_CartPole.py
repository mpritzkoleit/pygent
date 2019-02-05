from Environments import CartPole
from DDPG import DDPG
import numpy as np

def cost(x_, u_, x):
    x1, x2, x3, x4 = x_
    u1, = u_
    c = 1-np.cos(x2)
    if np.abs(x1)>= 1.:
        c = 100#(10*x1**8 + (1 - np.cos(x2)) + 1e-2*x3**2 + 1e-2*x4**2 + 1e-2*u1**2)
    return c


x0 = [np.random.uniform(-.5, .5), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
def x0fun():
    x0 = [np.random.uniform(-0.5, 0.5), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
    return x0

cartPole = CartPole(cost, x0fun)
t = 15
dt = 0.03

path = '../Results/DDPG/cartPole/'

algorithm = DDPG(cartPole, t, dt, path=path)
#algorithm.run_controller(x0fun())
#algorithm.animation()
#algorithm.plot()
#algorithm.run_episode()
algorithm.run_learning(1000)
