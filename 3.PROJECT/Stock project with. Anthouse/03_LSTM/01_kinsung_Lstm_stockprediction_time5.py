'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os
import pandas.io.data as web
import datetime
from pandas import read_csv,Series, DataFrame
import matplotlib.pyplot as plt
# matplotlib inline
from sklearn.preprocessing import MinMaxScaler

tf.set_random_seed(777)  # reproducibility
# np.set_printoptions(threshold=np.inf)

class KOSPIDATA:

    def __init__(self):
        start = datetime.datetime(1998, 5, 1)
        end = datetime.datetime(2016, 12, 31)
        kospi = web.DataReader("AAPL", "google", start, end)

        self.arr_date= np.array(kospi.index)
        self.arr_open = np.array(kospi.Open, dtype=float)
        self.arr_close= np.array(kospi.Close, dtype=float)
        self.arr_high= np.array(kospi.High, dtype=float)
        self.arr_low= np.array(kospi.Low, dtype=float)
        self.arr_volume= np.array(kospi.Volume, dtype=float)

# train Parameters
seq_length = 5
data_dim = 5
hidden_dim = 100
output_dim = 1
learning_rate = 0.01
iterations = 5000

K = KOSPIDATA()

FEATURES = ['high', 'low', 'open', 'close', 'volume']
# FEATURES = ['high', 'low', 'open', 'close']

data = {'year': K.arr_date,
        'high': K.arr_high,
        'low': K.arr_low,
        'open': K.arr_open,
        'close': K.arr_close,
        'volume' : K.arr_volume}

# df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close'])
df = DataFrame(data, columns=['year','high','low','open','close','volume'])

xy = np.array(df[FEATURES].values[:])
scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(xy)
pre_y = scaler.fit_transform(np.array(df['close'].values))
y = np.reshape(pre_y,(len(pre_y),1))

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])
x_index = df.year[train_size:len(dataX)]

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# MAE/RMSE/MAPE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
mae = tf.reduce_mean(tf.abs(targets - predictions))
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
mape = 100*tf.reduce_mean(tf.abs((targets-predictions)/targets))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    mae_val = sess.run(mae, feed_dict={targets: testY, predictions: test_predict})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    mape_val = sess.run(mape, feed_dict={targets: testY, predictions: test_predict})

    print("MAE: {}".format(mae_val))
    print("RMSE: {}".format(rmse_val))
    print("MAPE: {}".format(mape_val))

    new_y = scaler.inverse_transform(testY)
    new_pred_y = scaler.inverse_transform(test_predict)

    # Plot predictions

    #     plt.plot(testY) -> 스케일 조정 전 아웃풋
    #     plt.plot(test_predict)

    plt.plot(x_index, new_y, label="True Price")
    plt.plot(x_index, new_pred_y, label="Predicted Price")
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()