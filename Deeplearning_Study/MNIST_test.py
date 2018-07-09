import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        self.w1 = np.random.uniform(low=-0.01, high=0.01, size=(784,100))
        self.w2 = np.random.uniform(low=-0.01, high=0.01, size=(100,10))
        self.learning_rate = 0.01

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def der_sigmoid(self,x):
        return x*(1-x)

    def cross_entropy(self, y_predicted, y_label):
        return np.mean(-np.sum(y_label * np.log(self.softmax(y_predicted)),axis=1))

    def softmax(self, x):
        if x.ndim == 1:
            x = x.reshape([1, x.size])
        mod_x = x - np.max(x, 1).reshape([x.shape[0],1])
        sigmoid = np.exp(mod_x)
        return sigmoid / np.sum(sigmoid, axis=1).reshape([sigmoid.shape[0],1])

    def forward_propa(self,x):
        y1 = np.dot(x, self.w1)
        sigmoidy1 = self.sigmoid(y1)
        y2 = np.dot(sigmoidy1, self.w2)
        softmaxy2 = self.softmax(y2)
        return sigmoidy1, softmaxy2

    def back_propa(self,x,y_label,y_predicted1,y_predicted2):
        error = y_predicted2 - y_label
        dy2 = np.matmul(error, self.w2.T)
        dy1 = self.der_sigmoid(y_predicted1)
        dw1 = x.T.dot(dy2 * dy1)
        dw2 = y_predicted1.T.dot(error)
        return dw1, dw2

    def update(self, dw1, dw2):
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2


if __name__ =='__main__':

    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
    np.random.seed(777)
    NN = NN()
    loss_fun = []
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        y1, y2 = NN.forward_propa(batch_xs)
        dw1, dw2 = NN.back_propa(batch_xs,batch_ys,y1,y2)
        NN.update(dw1, dw2)
        y1, y2 = NN.forward_propa(mnist.test.images)
        correct = np.equal(np.argmax(y2,1), np.argmax(mnist.test.labels, 1))
        accuracy = np.mean(correct)
        loss = NN.cross_entropy(y2, mnist.test.labels)
        print('{0} accuracy: {1}, loss: {2}'.format(_, accuracy, loss))
        loss_fun.append(loss)

    plt.plot(loss_fun)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.show()




