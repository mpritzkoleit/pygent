from pygent.environments import CartPoleTriple
from pygent.agents import Agent
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u, t, mod):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    u1, = u
    c = 5*x1**2 + 1*x2**2 + 0.5*x3**2 + 0.5*x4**2
    c += 0.02*x5**2 + 0.05*x6**2 + 0.05*x7**2 + 0.05*x8**2 + 0.1*u1**2
    return c

# define the final cost at step N
def c_N(x, mod):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    c = 150*(x1-0)**2 #+ 100*(x2-mod.pi)**2 + 100*(x3-mod.pi)**2 + 100*(x4-mod.pi)**2 + 100*x5**2 + 100*x6**2 + 100*x7**2 + 100*x8**2
    return c

# initial state value
x0 = [.5, np.pi, np.pi, np.pi, 0, 0, 0, 0]

t = 10 # simulation time
dt = 0.002 # time step-size

agent = Agent(1)
env = CartPoleTriple(c_k, x0, dt)
env.uMax = 25

path = '../../../results/ext/triple_smooth/'  # path, where results are saved


alpha = 1e-3 #1/(1*dt + 1)
f = 150
shift = 0
a1 = 10
a2 = a1
k = -100
int_output = 0.
int_input_old = 0.
hp_filter = 0.
J = -c_N(env.x, np)*dt
J_old = 1.*J

for t in np.arange(0, t, dt):
    J = -c_N(env.x, np)*dt
    hp_filter = alpha*(hp_filter + dt*(J-J_old))
    int_input = hp_filter*a1*np.sin(2*np.pi*f*t + shift)
    #int_output = 1/(1-0.01*dt)*(50*(int_input-int_input_old) - 200*dt*int_input + int_output)
    int_output += dt*int_input*k
    u = [int_output + a2*np.sin(2*np.pi*f * t)]
    env.step(u)
    agent.control(dt, u)
    if env.terminated:
        break
    J_old = J
    int_input_old = int_input
    pass
print(str(env.x))
env.plot()
agent.plot()
plt.show()