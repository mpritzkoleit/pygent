__name__ == "pygent.data"

# todo: add dataset to init

import random
random.seed(0) #, the number of the beast!
import numpy as np 
np.random.seed(0)
import pickle

class DataSet(object):
    """ Data set to store data. First in, first out, when maximum capacity is reached.

    Attributes:
        data (list): data set of samples
        length (int): maximum capacity

    """

    def __init__(self, length):
        self.data = list()
        self.length = length

    def add_sample(self, sample):
        """ Adds sample to data set

        Args:
            sample (dict/tuple/array)

        Returns:
            exists (bool): True if sample is already in data set.

        """

        # if sample not in data, add to data
        if sample not in self.data:
            self.data.append(sample)
            exists = False
        else:
            exists = True

        # if data set exceeds length, pop first value (FIFO)
        if self.length < self.data.__len__():
            del self.data[0]
        return exists

    def force_add_sample(self, sample):
        """ Adds sample to data set without checking for duplicates."""

        self.data.append(sample)
        return True

    def rm_sample(self, sample):
        """ Removes sample from data set

        Args:
            sample (tuple)

        Returns:
            exists (bool): True if sample is in data set

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
        """ Samples randomly from the data set.

            Returns:
                sample
            """
        sample = np.random.choice(self.data)

        return sample

    def minibatch(self, n):
        """ Samples a minibatch of size 'n' from the data set.

            Args:
                n(int): size of the minibatch

            Returns:
                minibatch (list): batch of 'n' samples
            """

        #minibatch = random.sample(self.data, min(n, len(self.data)))
        minibatch = np.random.choice(self.data, min(n, len(self.data)))
        return minibatch

    def save(self, path):
        """ Save data in file.

            Args:
                path (string): 'directory/filename.p'
            """

        pickle.dump(self.data, open(path, 'wb'))
        pass

    def load(self, path):
        """ Load data from file.

            Args:
                path (string): 'directory/filename.p'
            """
        self.data = pickle.load(open(path, 'rb'))
        pass
