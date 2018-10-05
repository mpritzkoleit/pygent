import numpy as np
from Agents import Agent
from Data import DataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import abstractmethod


class MLP(nn.Module):
    """ Multilayer perceptron (MLP) with sigmoid activation functions implemented in PyTorch

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
        for i in range(self.nLayers - 1):
            layer = getattr(self, 'layer'+str(i))
            x = F.sigmoid(layer(x))
        return x

    def num_flat_features(self, x):
        return self.netStructure[0]
