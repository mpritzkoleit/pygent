from NeuralNetworkModels import MLP
import torch
from NFQ import QNetwork
network = MLP([5, 5, 5, 1])

x = torch.Tensor([1, 2, 3, 4, 5])
print(network.num_flat_features(x))
print(network.forward(x))

qNetwork = QNetwork([-1,0, 1], [4, 10, 10, 1])

