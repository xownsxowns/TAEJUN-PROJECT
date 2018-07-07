import MLP
from load import *

mnist = MLP.NN()
training_data, validation_data, test_data = load_mnist()
mnist.fit(training_data)
