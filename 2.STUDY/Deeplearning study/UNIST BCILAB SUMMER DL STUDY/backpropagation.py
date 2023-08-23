# XOR problem
# Backward propagation
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

    def dif_sigmoid(self, s):
        return s*(1-s)

    def backwardpropagation(self,x,y,output):
        self.o_error = y-output # error in output
        self.o_delta = self.o_error * self.dif_sigmoid(output)
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.dif_sigmoid(self.z2)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self, x, y):
        output = self.forward(x)
        self.backwardpropagation(x,y,output)

NN = neural_network()
for i in range(500):
    print("input: \n" + str(x))
    print("actual output: \n" + str(y))
    print("predicted output: \n" + str(NN.forward(x)))
    print("loss: \n" + str(np.mean(np.square(y-NN.forward(x)))))
    print("\n")
    NN.train(x,y)
