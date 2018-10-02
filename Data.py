import numpy as np
# import tensorflow as tf
from abc import abstractmethod, abstractproperty
# import scipy.integrate as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

class DataSet(object):
    """ DataSet to store data

    Attributes:
        data (set): data set to store tuples

    """

    def __init__(self, length):
        self.data = list()
        self.length = length

    def add_sample(self, sample):
        """ Adds sample to data

        Args:
            sample (tuple)

        Returns:
            exists (bool): True if sample is already in data

        """

        # if sample not in data, add to data
        if sample not in self.data:
            self.data.append(sample)
            exists = False
        else:
            exists = True

        # if data set exceeds lenght, pop first value (FIFO)
        if self.length < len(self.data):
            self.data.pop(0)

        return exists

    def add_sample(self, sample):
        """ Adds sample to data

        Args:
            sample (tuple)

        Returns:
            exists (bool): True if sample is already in data

        """

        # if sample not in data, add to data
        if sample not in self.data:
            self.data.append(sample)
            exists = False
        else:
            exists = True

        # if data set exceeds lenght, pop first value (FIFO)
        if self.length < len(self.data):
            self.data.pop(0)

        return exists

    def random_sample(self):
        sample = random.choice(self.data)
        return sample

    def minibatch(self, n):
        batch = random.sample(self.data, n)
        return batch