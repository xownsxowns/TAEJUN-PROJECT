import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# 연산 그래프가 실행될 때 제공되어야 하는 값, [None, 784] > 한 이미지의 사이즈가 784이며 얼마나 많은 이미지를 한 번에 사용할지는 이 시점에 지정하지 않겠다 (None)
x = tf.placeholder(tf.float32, [None, 784])
# variable은 연산과정에서 조작되는 값
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

# 경사하강법 사용, Learning rate = 0.5
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # 모든 변수 초기화
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))