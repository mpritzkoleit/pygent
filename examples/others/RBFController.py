import torch
from NeuralNetworkModels import  RBFController

ctr = RBFController(xDim=2, uDim=1)

x = torch.tensor([[1., 2.]])

RBFController(x)