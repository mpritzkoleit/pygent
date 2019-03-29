from Environments import Acrobot
import matplotlib.pyplot as plt
import numpy as np

def cost(x_, u_, x):
    return 0


x0 = [0.5*np.pi, 0, 0, 0]

cartPole = Acrobot(cost, x0)
t = 150
dt = 0.01

for _ in range(int(t/dt)):
    cartPole.step(dt, [0])

cartPole.plot()
plt.show()
cartPole.animation()