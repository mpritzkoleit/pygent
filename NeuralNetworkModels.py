import numpy as np
from Agents import Agent
from Data import DataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import abstractmethod


class MLP(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch

    Attributes:
        netStructure (array): layer structure of MLP: [1, 5, 5, 1] (2 hidden layer with 5 neurons, 1 input, 1 output)

    """

    def __init__(self, netStructure):
        super(MLP, self).__init__()
        self.netStructure = netStructure
        self.nLayers = len(netStructure)
        # create linear layers y = Wx + b

        for i in range(self.nLayers - 1):
            setattr(self, 'layer'+str(i), nn.Linear(netStructure[i], netStructure[i+1]))

    def forward(self, x):
        # connect layers
        for i in range(self.nLayers - 2):
            layer = getattr(self, 'layer'+str(i))
            x = F.tanh(layer(x))
        layer = getattr(self, 'layer' + str(self.nLayers-2))
        x = F.sigmoid(layer(x)) + F.sigmoid(layer(x))
        return x

class Critic(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch

    Attributes:
        netStructure (array): layer structure of MLP: [1, 5, 5, 1] (2 hidden layer with 5 neurons, 1 input, 1 output)

    """

    def __init__(self, xDim, uDim):
        super(Critic, self).__init__()
        self.xDim = xDim
        self.uDim = uDim
        self.layer1 = nn.Linear(xDim, 400)
        self.layer1_bn = nn.BatchNorm1d(400)
        self.layer21 = nn.Linear(400+uDim, 300)
        self.layer22 = nn.Linear(uDim, 300, bias=False) # Ax+Bu+b
        self.layer3 = nn.Linear(300, 1)

        # weight initialization
        wMin = -3*10e-3
        wMax = 3*10e-3
        self.layer1.weight = torch.nn.init.uniform_(self.layer1.weight, a=wMin, b=wMax)
        self.layer1.bias = torch.nn.init.uniform_(self.layer1.bias, a=wMin, b=wMax)
        self.layer21.weight = torch.nn.init.uniform_(self.layer21.weight, a=wMin, b=wMax)
        self.layer21.bias = torch.nn.init.uniform_(self.layer21.bias, a=wMin, b=wMax)
        self.layer22.weight = torch.nn.init.uniform_(self.layer22.weight, a=wMin, b=wMax)
        self.layer3.weight = torch.nn.init.uniform_(self.layer3.weight, a=wMin, b=wMax)
        self.layer3.bias = torch.nn.init.uniform_(self.layer3.bias, a=wMin, b=wMax)

    def forward(self, x, u):
        # connect layers
        h1 = F.relu(self.layer1_bn(self.layer1(x)))
        h2 = F.relu(self.layer21(torch.cat((h1, u), 1))) #+ self.layer22(u))
        y = F.relu(self.layer3(h2))
        return y



class Actor(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch

    Attributes:
        netStructure (array): layer structure of MLP: [1, 5, 5, 1] (2 hidden layer with 5 neurons, 1 input, 1 output)

    """

    def __init__(self, xDim, uDim, uMax=1.0):
        super(Actor, self).__init__()
        self.xDim = xDim
        self.uDim = uDim
        self.uMax = uMax
        self.layer1 = nn.Linear(xDim, 400)
        self.layer1_bn = nn.BatchNorm1d(400)
        self.layer2 = nn.Linear(400, 300)
        self.layer2_bn = nn.BatchNorm1d(300)
        self.layer3 = nn.Linear(300, uDim)

        # weight initialization
        wMin = -3*10e-3
        wMax = 3*10e-3
        self.layer1.weight = torch.nn.init.uniform_(self.layer1.weight, a=wMin, b=wMax)
        self.layer1.bias = torch.nn.init.uniform_(self.layer1.bias, a=wMin, b=wMax)
        self.layer2.weight = torch.nn.init.uniform_(self.layer2.weight, a=wMin, b=wMax)
        self.layer2.bias = torch.nn.init.uniform_(self.layer2.bias, a=wMin, b=wMax)
        self.layer3.weight = torch.nn.init.uniform_(self.layer3.weight, a=wMin, b=wMax)
        self.layer3.bias = torch.nn.init.uniform_(self.layer3.bias, a=wMin, b=wMax)

    def forward(self, x):
        # connect layers
        h1 = F.relu(self.layer1_bn(self.layer1(x)))
        h2 = F.relu(self.layer2_bn(self.layer2(h1)))
        y = F.tanh(self.layer3(h2))
        y = torch.mul(y, self.uMax) # scale output
        return y
