import numpy as np
import Agent
from Data import DataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import abstractmethod


class MLP(nn.Module):
    """ Multilayer perceptron (MLP) implementation in PyTorch

    Attributes:
        layers (int): layer structure of MLP

    """

    def __init__(self, netStructure):
        super(MLP, self).__init__()
        self.netStructure = netStructure
        self.nLayers = len(netStructure)

        # create linear layers y = Wx + b
        for i in range(self.nLayers - 1):
            setattr(self, 'layer'+str(i), nn.Linear(netStructure[i], netStructure[i+1]))

    def forward(self, x):
        # unfold neural network
        for i in range(self.nLayers - 1):
            layer = getattr(self, 'layer'+str(i))
            x = F.sigmoid(layer(x))
        return x

    def num_flat_features(self, x):
        return self.netStructure[0]
