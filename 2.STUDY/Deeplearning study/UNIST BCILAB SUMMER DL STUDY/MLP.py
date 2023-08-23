import os
import numpy as np
import random

from activation import sigmoid, sigmoid_prime

class NN(object):
    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=20, epochs=10):
        ''' size: list, optional
                A list of integers specifying number of neurons in each layer.
                Not required if a pretrained model is used'''
        self.sizes = sizes
        self.num_layers = len(sizes)

        # first term corresponds to layer 0 )input layer).
        # No weights enter the input layer and hence self.weights[0] is redundant
        self.weights = [np.array([0])] + [np.random.randn(y,x) for y, x in list(zip(sizes[1:], sizes[:-1]))]
        # input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y,1) for y in sizes]
        # Input layer has no weights, biases associated. Hence z=wx+b is not defined for input layer.
        # self.zs[0] is redundant
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # Training examples can be treated as activations coming out of input layer
        # Hence self.activations[0] = (training_example)
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = learning_rate

    def fit(self, training_data, validation_data = None):
        ''' Fitting is carried out using Stochastic Gradient Descent Algorithm
        parameter
        -----------------
        training_data : list of tuple
            A list of tuples of numpy arrays, ordered as (image, label)
        validation_data : list of tuple, optional '''
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            for mini_batch in mini_batches:
                bb = [np.zeros(bias.shape) for bias in self.biases]
                ww = [np.zeros(weight.shape) for weight in self.weights]
                for x,y in mini_batch:
                    self._forward_prop(x)
                    delta_bb_b, delta_ww_w = self._back_prop(x,y)
                    bb = [nb + dnb for nb, dnb in list(zip(bb, delta_bb_b))]
                    ww = [nw + dnw for nw, dnw in list(zip(ww, delta_ww_w))]
                self.weights = [
                    w - (self.eta / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, bb)]
            if validation_data:
                accuracy = self.validate(validation_data) / 100.0
                print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
            else:
                print("Processed epoch {0}.".format(epoch))

    def validate(self, validation_data):
        '''Validate the Neural Network on provided validation data. It uses the
        number of correctly predicted examples as validation accuracy metric.'''
        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        return sum(result for result in validation_results)

    def predict(self, x):
        '''Predict the label of a single test example (image).'''
        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self,x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i-1]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])

    def _back_prop(self,x,y):
        bb = [np.zeros(bias.shape) for bias in self.biases]
        ww = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * sigmoid_prime(self._zs[-1])
        bb[-1] = error
        ww[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l+1].transpose().dot(error),
                sigmoid_prime(self._zs[l])
            )
            bb[l] = error
            ww[l] = error.dot(self._activations[l-1].transpose())

        return bb, ww
