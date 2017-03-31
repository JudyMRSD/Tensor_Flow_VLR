""" MNIST Tutorial (ConvNet)
This tutorial guides you through a classification task on MNIST dataset
by constructing and trainnig a ConvNet.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# read training data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define input
x = tf.placeholder(tf.float32, shape=[None, 784])   # None tells TensorFlow that the dimension does not matter
x_image = tf.reshape(x, (-1, 28, 28, 1))            # the convention is TensorFlow is (b, h, w, c)

# build our network
# conv1 + relu + maxpool
W_conv1 = weight_variable([5, 5, 1, 32])            # the convention is (h, w, c_in, c_out)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv2 + relu + maxpool
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fc1 + relu
W_fc1 = weight_variable((7 * 7 * 64, 1024))
b_fc1 = bias_variable((1024,))

h_pool2_flat = tf.reshape(h_pool2, (-1, 7 * 7 * 64))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout (prevent over-fitting)
# set keep_prob as a input so we can disable dropout layer during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# read-out (fc2)
W_fc2 = weight_variable((1024, 10))
b_fc2 = bias_variable((10,))
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cross-entropy loss
y_gt = tf.placeholder(tf.int32, shape=[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
precision = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# start trining
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# summaries for tensor board
tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('precision', precision)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('tensor_logs', sess.graph)

for i in range(20000):
    batch = mnist.train.next_batch(50)

    _, summary = sess.run([train, summary_op], feed_dict={x: batch[0], y_gt: batch[1], keep_prob: 0.5})
    summary_writer.add_summary(summary, global_step=i)

# evaluating on test set
performance = sess.run(precision, feed_dict={x: mnist.test.images, y_gt: mnist.test.labels, keep_prob: 1.0})
print('precision on test set: {0}'.format(performance))


