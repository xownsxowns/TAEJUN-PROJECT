# Forward Propagation, Backpropagation
# XOR problem
import numpy as np

x = np.array(([0,0],[1,0],[0,1],[1,1]), dtype=float)
y = np.array(([0],[1],[1],[0]), dtype=float)

class neural_network(object):
    def __init__(self):
        self.inputsize = 2
        self.outputsize = 1
        self.hiddensize = 2

        # weight initialization
        self.w1 = np.random.randn(self.inputsize, self.hiddensize)
        self.w2 = np.random.randn(self.hiddensize, self.outputsize)

