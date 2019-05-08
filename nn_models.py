import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import fanin_init

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
            x = torch.tanh(layer(x))
        layer = getattr(self, 'layer' + str(self.nLayers-2))
        x = torch.tanh(layer(x))
        return x

class CriticBN(nn.Module):

    def __init__(self, xDim, uDim):
        super(CriticBN, self).__init__()
        self.xDim = xDim
        self.uDim = uDim
        self.layer0_bn = nn.BatchNorm1d(xDim)
        self.layer1 = nn.Linear(xDim, 400)
        self.layer1_bn = nn.BatchNorm1d(400)
        self.layer2 = nn.Linear(400 + uDim, 300)
        self.layer3 = nn.Linear(300, 1)

        # weight initialization
        wMin = -3.0*1e-3
        wMax = 3.0*1e-3
        fanin_init(self.layer1)
        fanin_init(self.layer2)
        self.layer3.weight = torch.nn.init.uniform_(self.layer3.weight, a=wMin, b=wMax)
        self.layer3.bias = torch.nn.init.uniform_(self.layer3.bias, a=wMin, b=wMax)

    def forward(self, x, u):
        # connect layers
        x_bn = self.layer0_bn(x)
        h1_bn = self.layer1_bn(self.layer1(x_bn))
        h1 = F.relu(h1_bn)
        h2_in = torch.cat((h1, u), 1)
        h2 = self.layer2(h2_in)
        h2_out = F.relu(h2)
        y = self.layer3(h2_out)
        return y

class Critic(nn.Module):
    def __init__(self, xDim, uDim):
        super(Critic, self).__init__()
        self.xDim = xDim
        self.uDim = uDim
        self.layer1 = nn.Linear(xDim, 400)
        self.layer2 = nn.Linear(400 + uDim, 300)
        self.layer3 = nn.Linear(300, 1)

        # weight initialization
        wMin = -3.0*1e-3
        wMax = 3.0*1e-3
        fanin_init(self.layer1)
        fanin_init(self.layer2)
        self.layer3.weight = torch.nn.init.uniform_(self.layer3.weight, a=wMin, b=wMax)
        self.layer3.bias = torch.nn.init.uniform_(self.layer3.bias, a=wMin, b=wMax)

    def forward(self, x, u):
        # connect layers
        h1 = F.relu(self.layer1(x))
        h2_in = torch.cat((h1, u), 1)
        h2 = self.layer2(h2_in)
        h2_out = F.relu(h2)
        y = self.layer3(h2_out)
        return y

class ActorBN(nn.Module):
    def __init__(self, xDim, uDim, uMax=1.0):
        super(ActorBN, self).__init__()
        self.xDim = xDim
        self.uDim = uDim
        self.uMax = uMax
        self.layer0_bn = nn.BatchNorm1d(xDim)
        self.layer1 = nn.Linear(xDim, 400)
        self.layer1_bn = nn.BatchNorm1d(400)
        self.layer2 = nn.Linear(400, 300)
        self.layer2_bn = nn.BatchNorm1d(300)
        self.layer3 = nn.Linear(300, uDim)

        # weight initialization
        wMin = -3*1e-3
        wMax = 3*1e-3
        fanin_init(self.layer1)
        fanin_init(self.layer2)
        self.layer3.weight = torch.nn.init.uniform_(self.layer3.weight, a=wMin, b=wMax)
        self.layer3.bias = torch.nn.init.uniform_(self.layer3.bias, a=wMin, b=wMax)

    def forward(self, x):
        x_bn = self.layer0_bn(x)
        h1_bn = self.layer1_bn(self.layer1(x_bn))
        h1 = F.relu(h1_bn)
        h2_bn = self.layer2_bn(self.layer2(h1))
        h2 = F.relu(h2_bn)
        h3 = self.layer3(h2)
        y = torch.tanh(h3)
        y = torch.mul(y, self.uMax) # scale output
        return y

class Actor(nn.Module):
    def __init__(self, xDim, uDim, uMax=1.0):
        super(Actor, self).__init__()
        self.xDim = xDim
        self.uDim = uDim
        self.uMax = uMax
        self.layer1 = nn.Linear(xDim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, uDim)

        # weight initialization
        wMin = -3*1e-3
        wMax = 3*1e-3
        fanin_init(self.layer1)
        fanin_init(self.layer2)
        self.layer3.weight = torch.nn.init.uniform_(self.layer3.weight, a=wMin, b=wMax)
        self.layer3.bias = torch.nn.init.uniform_(self.layer3.bias, a=wMin, b=wMax)

    def forward(self, x):
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = torch.tanh(self.layer3(h2))
        y = torch.mul(h3, self.uMax) # scale output
        return y

