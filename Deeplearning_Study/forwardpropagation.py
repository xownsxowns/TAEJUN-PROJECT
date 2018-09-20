# XOR problem
# Forward propagation
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

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def forward(self, x):
        # forward propagation through our network
        self.z1 = np.dot(x, self.w1)
        self.z2 = self.sigmoid(self.z1)
        self.z3 = np.dot(self.z2, self.w2)
        output = self.sigmoid(self.z3)
        return output

NN = neural_network()
output = NN.forward(x)
print(output)
