# pygent

This repository arised out of a diploma thesis at Institute of Control Theory, Dresden University of Technology, Germany. The software led to several publications where it is cited. More information and additional material regarding the the publications can be found in the respective branches:

- [`ifac`](../../tree/ifac): "Reinforcement Learning and Trajectory Planning based on Model Approximation with Neural Networks applied to Transition Problems"
- [`gma`](../../tree/gma): "Bestärkendes Lernen mittels Offline-Trajektorienplanung basierend auf iterativ approximierten Modellen"
- [`at-beitrag`](../../tree/at-beitrag): "Bestärkendes Lernen mittels Offline-Trajektorienplanung basierend auf iterativ approximierten Modellen"

Thesis: Bestärkendes Lernen zur Steuerung und Regelung nichtlinearer dynamischer Systeme (engl. Reinforcement Learning for the Control of Nonlinear Dynamical Systems)

Link: https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-377219 (written in German)
---


The goal of the thesis was to investigate the state-of-the-art in reinforcement learning for continuous control of nonlinear dynamic systems

As a reference method, iLQR is implemented, a trajectory optimization algorithm referenced in many RL papers, to solve such problems in a model based fashion.

Installation:

  clone or download the package

  run: python setup.py install

If you want to reproduce the results of the paper 'at-beitrag', please run the examples 'cart_pole_hpc.py' and 'pendulum_hpc.py'.