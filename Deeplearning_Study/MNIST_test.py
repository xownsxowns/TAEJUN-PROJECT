import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class NN:
    def __init__(self):
        self.w1 = np.random.uniform(low=-0.01, high=0.01, size=(784,100))
        self.w2 = np.random.uniform(low=-0.01, high=0.01, size=(100,10))
        self.learning_rate = 0.01

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def der_sigmoid(self,x):
        return x*(1-x)

    def softmax(self, y_predicted, y_label):
        pass

    def cross_entropy(self, x):
        pass

    def forward_propa(self,x):
        pass

    def back_propa(self,x,y_label,y_predicted1,y_predicted2):
        pass

    def update(self, dw1, dw2):
        pass

