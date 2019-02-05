import numpy as np
# import tensorflow as tf
from abc import abstractmethod, abstractproperty
# import scipy.integrate as sci
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import pickle

class DataSet(object):
    """ DataSet to store data

    Attributes:
        data (set): data set to store tuples

    """

    def __init__(self, lengthMax):
        self.data = list()
        self.length = lengthMax

    def add_sample(self, sample):
        """ Adds sample to data

        Args:
            sample (dict/tuple/array)

        Returns:
            exists (bool): True if sample is already in data

        """

        # if sample not in data, add to data
        if sample not in self.data:
        #if not any([sample in self.data[_] for _ in range(len(self.data))]):
            self.data.append(sample)
            exists = False
        else:
            exists = True

        # if data set exceeds length, pop first value (FIFO)
        if self.length < len(self.data):
            del self.data[0]

        return exists

    def rm_sample(self, sample):
        """ Removes sample to data

        Args:
            sample (tuple)

        Returns:
            exists (bool): True if sample is in data

        """

        # if sample not in data, add to data
        if sample in self.data:
            # find idx of sample in self.data
            self.data.remove(sample)
            exists = True
        else:
            exists = False

        return exists

    def random_sample(self):
        sample = random.choice(self.data)
        return sample

    def minibatch(self, n):
        batch = random.sample(self.data, min(n, len(self.data)))
        return batch

    def save(self, path):
        pickle.dump(self.data, open(path, 'wb'))

    def load(self, path):
        self.data = pickle.load(open(path, 'rb'))

