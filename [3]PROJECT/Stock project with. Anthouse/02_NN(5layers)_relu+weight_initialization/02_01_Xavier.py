import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from pandas import Series, DataFrame
import sklearn
from sklearn.cross_validation import train_test_split

start = datetime.datetime(1993,2,19)
end = datetime.datetime(2016,5,4)

gs = web.DataReader("078930.KS", "yahoo", start, end)

frame = DataFrame(gs, columns=['Open', 'High', 'Low', 'Close', 'Volume','Adj Close'])

def prediction():
    profit = []
    for i in range(len(frame['Adj Close']) - 1):
        if frame['Adj Close'][i] < frame['Adj Close'][i + 1]:
            profit.append('1')
        else:
            profit.append('0')
    return profit

profit = np.array(prediction())
profit = np.append(profit, np.NaN)
new_X2 = profit[:-1]
new_profit = np.reshape(new_X2, [2979,1])
print(profit)
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume','Adj Close']
new_X = np.array(frame[FEATURES].values[:-1,:])

train, test = train_test_split(new_X, test_size = 0.2)

print(train)
print(test)
print(new_X)
print(new_X.shape)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.get_variable('w1_xavier',[6,30], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('w2_xavier',[30,24], initializer = tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('w3_xavier',[24,12], initializer = tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable('w4_xavier',[12,8], initializer = tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable('w5_xavier',[8,2], initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([30]), name="Bias1")
b2 = tf.Variable(tf.zeros([24]), name="Bias2")
b3 = tf.Variable(tf.zeros([12]), name="Bias3")
b4 = tf.Variable(tf.zeros([8]), name="Bias3")
b5 = tf.Variable(tf.zeros([2]), name="Bias3")

L2 = tf.nn.relu(tf.matmul(X,W1)+b1)
L3 = tf.nn.relu(tf.matmul(L2,W2)+b2)
L4 = tf.nn.relu(tf.matmul(L3,W3)+b3)
L5 = tf.nn.relu(tf.matmul(L4,W4)+b4)

hypothesis = tf.nn.relu(tf.matmul(L5,W5)+b5)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1001):
        sess.run(train, feed_dict = {X:new_X, Y:new_profit})
        if step % 200 == 0:
            step += 1
            print(step, sess.run(cost, feed_dict={X:new_X, Y:new_profit}), sess.run(W1), sess.run(W2), sess.run(W3),sess.run(W4),sess.run(W5))

    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:new_X, Y:new_profit}))
    print("Accuracy:", accuracy.eval({X:new_X, Y:new_profit}))
