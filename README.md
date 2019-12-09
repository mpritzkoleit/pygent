# pygent

This repository arised out of a diploma thesis at Institute of Control Theory, Dresden University of Technology, Germany. The software led to several publications where it is cited. More information and additional material regarding the the publications can be found in the respective branches:

- [`ifac`](../../tree/ifac): "Reinforcement Learning and Trajectory Planning based on Model Approximation with Neural Networks applied to Transition Problems"
- [`gma`](../../tree/gma): "Bestärkendes Lernen mittels Offline-Trajektorienplanung basierend auf iterativ approximierten Modellen"

Thesis: Application of Reinforcement Learning for the Control of Nonlinear Dynamic Systems


---

Paper: 

M. Pritzkoleit, C. Knoll, K. Röbenack - Reinforcement Learning and Trajectory Planning based on Model Approximation with Neural Networks applied to Transition Problems


Abstract:

In this paper we use a multilayer neural network to approximate the dynamics of nonlinear (mechanical) control systems. Furthermore, these neural network models are combined with offline trajectory planning, to form a model-based reinforcement learning (RL) algorithm, suitable for transition problems of nonlinear dynamical systems. We evaluate the algorithm on the swing-up of the cart-pole benchmark system and observe a significant performance gain in terms of data efficiency compared to a state-of-the-art model-free RL method (Deep Deterministic Policy Gradient (DDPG)). Additionally, we present first experimental results on a cart-triple-pole system test bench. For a simple transition problem, the proposed algorithm shows a good controller performance.


Installation:

  clone or download the package

  run: python setup.py install

To reproduce the simulation results of the paper, have a look at the  [`examples folder`](../../tree/ifac/examples).