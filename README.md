# pygent

This repository arised out of a diploma thesis on the topic "Application of Reinforcement Learning for the Control of Nonlinear Dynamic Systems" at Institute of Control Theory, Dresden University of Technology, Germany. The software led to several publications where this repo is cited. More information and additional material regarding the these publications can be found in the respective branches:

- [`ifac`](../../tree/ifac): "Reinforcement Learning and Trajectory Planning based on Model Approximation with Neural Networks applied to Transition Problems"
- [`gma`](../../tree/gma): "Bestärkendes Lernen mittels Offline-Trajektorienplanung basierend auf iterativ approximierten Modellen"


If you have any question or comment on this project or the regarding publications, do not hesitate to contact us:

- [Original author](https://github.com/mpritzkoleit)
- [Current corresponding author](https://tu-dresden.de/ing/elektrotechnik/rst/das-institut/beschaeftigte/carsten-knoll)


---


The goal of the thesis was to investigate the state-of-the-art in reinforcement learning for continuous control of nonlinear dynamic systems

As a reference method, iLQR is implemented, a trajectory optimization algorithm referenced in many RL papers, to solve such problems in a model based fashion.

Installation:

  clone or download the package

  run: python setup.py install

For now please have a look at the 'examples' folder.
