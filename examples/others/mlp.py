from NeuralNetworkModels import MLP, Critic, Actor
import torch
from NFQ import QNetwork
network = MLP([5, 5, 5, 1])

x = torch.Tensor([1, 2, 3, 4, 5])
print(network(x))

qNetwork = QNetwork([-1,0, 1], [4, 10, 10, 1], 0.0, 0.99, [0, 0, 0])

print(network.layer1.weight)

critic = Critic(4, 1)