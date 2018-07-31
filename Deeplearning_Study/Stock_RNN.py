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

    def loss(self, x, y):
        pass

    def bptt(self, x, y):
        pass

    def SGD(self, x, y, learning_rate):
        pass

    def train(self, x_train, y_train, learning_rate = 0.05, nepoch=100, eval_loss=5):
        pass


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
print(o.shape)
print(o)