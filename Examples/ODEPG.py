import torch
import torchdiffeq
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint#_adjoint as odeint
import matplotlib.pyplot as plt

from NeuralNetworkModels import Actor
from NeuralNetworkModels import MLP

from Environments import Pendulum

def cost(x_, u_, x):
    x1, x2 = x_
    u1, = u_
    c = (x1**2 + 1e-1*x2**2 + 1e-3*u1**2)
    return c

class Lambda(nn.Module):
    def __init__(self, controller):
        super(Lambda, self).__init__()
        self.controller = controller
        self.g = 9.81  # gravity
        self.b = 0.1  # dissipation


    def forward(self, t, y):
        x1, x2 = torch.split(y, 1, dim=1)
        u1 = 5*self.controller(y)
        dx1 = x2
        dx2 = u1 + self.g*torch.sin(x1) - self.b*x2
        return torch.cat((dx1, dx2)).t()

class Lambda2(nn.Module):
    def __init__(self, controller):
        super(Lambda2, self).__init__()
        self.controller = controller
        self.g = 9.81  # gravity
        self.b = 0.1  # dissipation

    def forward(self, t, y):
        x1, x2, x3, x4, x5, x6 = torch.split(y, 1, dim=1)
        u1 = self.controller(y)
        l1 = 0.5
        l2 = 0.7
        g = 9.81

        dx1 = x4
        dx2 = x5
        dx3 = x6
        dx4 = u1
        dx5 = 1 / l1 * (g * torch.sin(x2) + u1 * torch.cos(x2))
        dx6 = 1 / l2 * (g * torch.sin(x3) + u1 * torch.cos(x3))
        return torch.cat((dx1, dx2, dx3, dx4, dx5, dx6)).t()


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.layer1 = nn.Linear(2, 1)

    def forward(self, x):
        return torch.tanh(self.layer1(x))

#ctr = Actor(2, 1)
ctr = MLP([6, 20, 20, 1])
#ctr = Controller()
ode = Lambda2(ctr)

tf = 10.

t = torch.linspace(0., tf, 200)
def x0fun():
    x20 = np.random.uniform(3.14, 3.14)
    x0 = torch.tensor([[float(x20), 0.]])
    return x0

def x0fun2():
    x20 = np.random.uniform(3.14, 3.14)
    x0 = torch.tensor([[0., float(x20), float(x20), 0., 0., 0.]])
    return x0

#u0 = ctr(0)
#print(x0fun(), u0)

#print('ODE', ode(0, x0fun()))

xt_sample = odeint(ode, x0fun2(), t, method='dopri5')


optimizer = optim.Adam(ctr.parameters(), lr=1e-3)

for itr in range(1000):
    optimizer.zero_grad()
    xt_sample = odeint(ode, x0fun2(), t, method='dopri5')
    loss = torch.mean(torch.abs(xt_sample[:,0,0:3]))
    print('Iteration:', itr, ' Loss: ', loss.item())
    loss.backward()
    for _ in range(1):
        optimizer.step()
    if itr % 20 == 0:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(t.numpy(), xt_sample[:,0,:].detach().numpy())
        ax[1].plot(t.numpy(), ctr(xt_sample)[:,0,:].detach().numpy())
        plt.show()