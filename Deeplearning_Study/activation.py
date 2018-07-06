import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def relu(z):
    return np.maximum(z,0)

def relu_prime(z):
    return float(z > 0)