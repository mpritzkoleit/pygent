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

class CriticBN(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch

    Attributes:
        netStructure (array): layer structure of MLP: [1, 5, 5, 1] (2 hidden layer with 5 neurons, 1 input, 1 output)

    """

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
        y = F.relu(self.layer3(h2_out))
        return y

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
        y = F.relu(self.layer3(h2_out))
        return y

class ActorBN(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch

    Attributes:
        netStructure (array): layer structure of MLP: [1, 5, 5, 1] (2 hidden layer with 5 neurons, 1 input, 1 output)

    """

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
        y = F.tanh(h3)
        y = torch.mul(y, self.uMax) # scale output
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
        h3 = F.tanh(self.layer3(h2))
        y = torch.mul(h3, self.uMax) # scale output
        return y


class CriticDeep(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(CriticDeep, self).__init__()
        EPS = 3*1e-3

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim,256)
        fanin_init(self.fcs1)
        self.fcs2 = nn.Linear(256,128)
        fanin_init(self.fcs2)

        self.fca1 = nn.Linear(action_dim,128)
        fanin_init(self.fca1)

        self.fc2 = nn.Linear(256,128)
        fanin_init(self.fc2)

        self.fc3 = nn.Linear(128,1)
        self.fc3.weight.data.uniform_(-EPS,EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2,a1),dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ActorDeep(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(ActorDeep, self).__init__()
        EPS = 3 * 1e-3
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim,256)
        fanin_init(self.fc1)

        self.fc2 = nn.Linear(256,128)
        fanin_init(self.fc2)

        self.fc3 = nn.Linear(128,64)
        fanin_init(self.fc3)

        self.fc4 = nn.Linear(64,action_dim)
        self.fc4.weight.data.uniform_(-EPS,EPS)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))

        action = action #* self.action_lim

        return action

def fanin_init(layer):
    f = layer.in_features
    w_init = 1.0/np.sqrt(f)
    layer.weight = torch.nn.init.uniform_(layer.weight, a=-w_init, b=w_init)
    layer.bias = torch.nn.init.uniform_(layer.bias, a=-w_init, b=w_init)
    pass