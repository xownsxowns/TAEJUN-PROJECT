import tensorflow as tf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from pandas import Series, DataFrame
from sklearn import preprocessing

start = datetime.datetime(1998,5,1)
end = datetime.datetime(2016,5,4)

gs = web.DataReader("^KS11", "yahoo", start, end)

frame = DataFrame(gs, columns=['Open', 'High', 'Low', 'Close', 'Volume','Adj Close'])
new_frame = frame[frame['Volume'] != 0]

def prediction():
    profit = []
    for i in range(len(new_frame['Adj Close']) - 1):
        if new_frame['Adj Close'][i] < new_frame['Adj Close'][i + 1]:
            profit.append([1,0])
        else:
            profit.append([0,1])

    return profit

profit = np.array(prediction())
# profit = np.append(profit, np.NaN)

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
new_X = np.array(new_frame[FEATURES].values[:-1,:])
new_XX = preprocessing.scale(new_X)
print(new_XX)
print(profit.shape)

train_x = new_X[:3545, :]
train_y = profit[:3545, :]

test_x = new_X[3545:, :]
test_y = profit[3545:, :]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([6,12], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([12,2], -1.0, 1.0))
# W3 = tf.Variable(tf.random_uniform([24,12], -1.0, 1.0))
# W4 = tf.Variable(tf.random_uniform([12,8], -1.0, 1.0))
# W5 = tf.Variable(tf.random_uniform([8,2], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([12]), name="Bias1")
b2 = tf.Variable(tf.zeros([2]), name="Bias2")
# b3 = tf.Variable(tf.zeros([12]), name="Bias3")
# b4 = tf.Variable(tf.zeros([8]), name="Bias3")
# b5 = tf.Variable(tf.zeros([2]), name="Bias3")

L2 = tf.sigmoid(tf.matmul(X,W1)+b1)
# L3 = tf.sigmoid(tf.matmul(L2,W2)+b2)
# L4 = tf.sigmoid(tf.matmul(L3,W3)+b3)
# L5 = tf.sigmoid(tf.matmul(L4,W4)+b4)

# hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2) # No need to use softmax here
hypothesis = tf.matmul(L2, W2) + b2 # No need to use softmax here
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)) # Softmax loss
# cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
# Adam 쓰면 0 나오고, GD 쓰면 nan값이 졸라게 나오는데 accuracy는 0.540114나옴

a = tf.Variable(0.1)
optimizer = tf.train.AdamOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10001):
        sess.run(train, feed_dict = {X:train_x, Y:train_y})
        if step % 2000 == 0:
            step += 1
            print(step, sess.run(cost, feed_dict={X:train_x, Y:train_y}), sess.run(W1), sess.run(W2))

    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:train_x, Y:train_y}))
    print("Accuracy:", accuracy.eval({X:test_x, Y:test_y}))
