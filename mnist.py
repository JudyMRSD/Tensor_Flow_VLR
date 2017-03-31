#tutorial from https://www.tensorflow.org/get_started/mnist/pros
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

#--------------------initialize-----------------------------
#read training data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#x : input image 28x28
x = tf.placeholder(tf.float32, shape=[None, 784])
#target output class 
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#define weight and bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#Initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#Prediction and loss function
y = tf.matmul(x,W) + b
#Loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#train the model
#apply the gradient descent updates to the parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#--------------------training----------------------------

#Training the model can therefore be accomplished by repeatedly running train_step
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#--------------------Evaluate ----------------------------
#tf.argmax(y,1) is the label our model thinks is most likely for each input
#tf.argmax(y_,1) is the true label.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#convert boolean to 0, 1 , then take average 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Evaluate on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
























