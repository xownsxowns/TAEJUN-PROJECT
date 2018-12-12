import numpy as np

class RNN():
    def __init__(self, input_dim=5, hidden_dim=100, output_dim=1, bptt=4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt = bptt
        self.U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(hidden_dim, ))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim,hidden_dim))

    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    def propagation(self, x):
        T = len(x)
        # save all hidden states in a because need them later.
        h = np.zeros((T+1, self.hidden_dim))
        h[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. save them for later.
        o = np.zeros((T, self.output_dim))
        # For each time step...
        for t in np.arange(T):
            h[t] = np.tanh(np.dot(x[t], self.U) + self.W.dot(h[t-1]))
            o[t] = self.softmax(self.V.dot(h[t]))
        return [o, h]

    # def total_loss(self, x, y):
    #     L = 0
    #     for i in np.arange(len(y)):
    #         o, h = self.propagation(x[i])
    #         correct_pred = o[np.arange(len(y[i])), y[i]]
    #         L += -1 * np.sum(np.log(correct_pred))
    #     return L
    #
    # def bptt(self, x, y):
    #     T = len(y)
    #     o, h = self.propagation(x)
    #     dLdU = np.zeros(self.U.shape)
    #     dLdV = np.zeros(self.V.shape)
    #     dLdW = np.zeros(self.W.shape)
    #     delta_o = o
    #     delta_o[np.arange(len(y)), y] -= 1
    #     for t in np.arange(T)[::-1]:
    #         dLdV += np.outer(delta_o[t], h[t].T)
    #         delta_t = self.V.T.dot(delta_o[t]) * (1-(h[t] ** 2))
    #         # Backpropagation through time
    #         for bptt_step in np.arange(max(0, t-self.bptt), t+1)[::-1]:
    #             dLdW += np.outer(delta_t, h[bptt_step-1])
    #             dLdU[:, x[bptt_step]] += delta_t
    #             delta_t = self.W.T.dot(delta_t) * (1-h[bptt_step-1] ** 2)
    #         return [dLdU, dLdV, dLdW]
    #
    # def SGD(self, x, y, learning_rate):
    #     dLdU, dLdV, dLdW = self.bptt(x, y)
    #     self.U -= learning_rate * dLdU
    #     self.V -= learning_rate * dLdV
    #     self.W -= learning_rate * dLdW


import pandas as pd
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt

stock = pd.read_csv('Google.csv')
stock.Date = pd.to_datetime(stock.Date)
Features = ['Open','High','Low','Close','Volume']
data = {'Date':stock['Date'],'Open':stock['Adj. Open'], 'High':stock['Adj. High'], 'Low':stock['Adj. Low'], 'Close':stock['Adj. Close'], 'Volume':stock['Adj. Volume']}
Stock = pd.DataFrame(data, columns = ['Date','Open','High','Low','Close','Volume'])
new_x = np.array(Stock[Features])[:-1,] # 맨 마지막 데이터는 제거
x_data = preprocessing.scale(new_x)
y_data = np.roll(Stock['Close'][:].tolist(),-1)[:-1] # 다음날 종가로 데이터 한칸씩 땡기기

RNN = RNN()
o, h = RNN.propagation(x_data)


# def train(model, x_train, y_train, learning_rate=0.05, nepoch=100, eval_loss=5):
#     losses = []
#     num = 0
#     for epoch in range(nepoch):
#         if (epoch % eval_loss == 0):
#             loss = model.total_loss(x_train, y_train)
#             losses.append((num, loss))
#             print("Loss after num=%d epoch=%d: %f" % (num, epoch, loss))
#         for i in range(len(y_train)):
#             model.SGD(x_train[i], y_train[i], learning_rate)
#             num += 1
#
# np.random.seed(10)
# model = RNN()
# losses = train(model, x_data, y_data, nepoch=10, eval_loss=1)

