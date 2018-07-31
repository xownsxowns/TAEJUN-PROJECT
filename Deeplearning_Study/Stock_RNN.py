import numpy as np

class RNN():
    def __init__(self, input_dim, hidden_dim=100, bptt=4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt = bptt
        self.U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(hidden_dim, input_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim,hidden_dim))

    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    def propagation(self, x):
        pass

    def loss(self, x, y):
        pass

    def bptt(self, x, y):
        pass

    def SGD(self, x, y, learning_rate):
        pass

    def train(self, x_train, y_train, learning_rate = 0.05, nepoch=100, eval_loss=5):
        pass


