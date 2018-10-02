import numpy as np
import matplotlib.pyplot as plt
from Data import DataSet

x1 = [1, 2]
x2 = [2, 2]
x3 = [2, 3]

u1 = [1]
u2 = [2]

sample1 = ({'x_': x1, 'x': x2, 'u': u1})
sample2 = ({'x_': x2, 'x': x3, 'u': u2})
sample3 = (x2, x3, u2)

dataset = DataSet(100)

dataset.add_sample(sample1)
dataset.add_sample(sample2)
dataset.add_sample(sample1)
print(dataset.data)
print(dataset.data[0])
print(dataset.data[0]['x_'])
dataset.rm_sample(sample1)
print(dataset.data)