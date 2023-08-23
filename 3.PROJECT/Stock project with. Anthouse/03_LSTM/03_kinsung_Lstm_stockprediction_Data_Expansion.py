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
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import sqlite3

tf.set_random_seed(777)  # reproducibility
np.set_printoptions(threshold=np.inf)

class Indicator:
    def __init__(self):

        try:
            con = sqlite3.connect("02_KOSDAQ.db")
            cursor = con.cursor()
            cursor.execute("SELECT * FROM kosdaq_data")
            rows = cursor.fetchall()

            self.k_date, self.high_price, self.low_price, self.open_price, self.close_price, self.volume, self.turnover, self.sign, self.diff, self.foreigner_net, self.organ_net = [], [], [], [], [], [], [], [], [], [], []

            for row in rows:
                self.k_date.append(datetime.datetime.strptime("%s"%str(row[0]), "%Y%m%d").date())
                self.high_price.append(str(row[1]))
                self.low_price.append(str(row[2]))
                self.open_price.append(str(row[3]))
                self.close_price.append(str(row[4]))
                self.volume.append(str(row[5]))
                self.turnover.append(str(row[6]))
                self.sign.append(str(row[7]))
                self.diff.append(str(row[8]))
                self.foreigner_net.append(str(row[9]))
                self.organ_net.append(str(row[10]))

            self.k_date.reverse()
            self.high_price.reverse()
            self.low_price.reverse()
            self.open_price.reverse()
            self.close_price.reverse()
            self.volume.reverse()
            self.turnover.reverse()
            self.sign.reverse()
            self.diff.reverse()
            self.foreigner_net.reverse()
            self.organ_net.reverse()


            self.k_date = np.array(self.k_date)
            self.high_price = np.array(self.high_price)
            self.low_price = np.array(self.low_price)
            self.open_price = np.array(self.open_price)
            self.close_price = np.array(self.close_price)
            self.volume = np.array(self.volume)
            self.turnover = np.array(self.turnover)
            self.sign = np.array(self.sign)
            self.diff = np.array(self.diff)
            self.foreigner_net = np.array(self.foreigner_net)
            self.organ_net = np.array(self.organ_net)

        except:
            print("error database connection")

# train Parameters
seq_length = 5
data_dim = 7
hidden_dim = 7
output_dim = 1
learning_rate = 0.01
iterations = 5000

# FEATURES = ['k_date', 'high_price', 'low_price', 'open_price', 'close_price', 'volume', 'turnover', 'sign', 'diff', 'foreigner_net', 'organ_net']
FEATURES = ['high_price', 'low_price', 'open_price', 'close_price', 'volume', 'turnover', 'diff']

A = Indicator()
data = {'k_date': A.k_date,
        'high_price': A.high_price,
        'low_price': A.low_price,
        'open_price': A.open_price,
        'close_price': A.close_price,
        'volume': A.volume,
        'turnover': A.turnover,
        'sign': A.sign,
        'diff': A.diff,
        'foreigner_net': A.foreigner_net,
        'organ_net': A.organ_net
        }
df = DataFrame(data,
               columns=['k_date', 'high_price', 'low_price', 'open_price', 'close_price', 'volume', 'turnover',
                        'sign', 'diff', 'foreigner_net', 'organ_net'])

xy = np.array(df[FEATURES].values[:])
scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(xy)
pre_y = scaler.fit_transform(np.array(df['close_price'].values))
y = np.reshape(pre_y, (len(pre_y), 1))

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
x_index = df.k_date[train_size:len(dataX)]

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim, activation=tf.tanh)
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
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
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

    mae_val2 = sess.run(mae, feed_dict={targets: new_y, predictions: new_pred_y})
    rmse_val2 = sess.run(rmse, feed_dict={targets: new_y, predictions: new_pred_y})
    mape_val2 = sess.run(mape, feed_dict={targets: new_y, predictions: new_pred_y})

    print("MAE2: {}".format(mae_val2))
    print("RMSE2: {}".format(rmse_val2))
    print("MAPE2: {}".format(mape_val2))

    # Plot predictions

    #     plt.plot(testY) -> 스케일 조정 전 아웃풋
    #     plt.plot(test_predict)

    plt.plot(x_index, new_y, label="True Price")
    plt.plot(x_index, new_pred_y, label="Predicted Price")
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()