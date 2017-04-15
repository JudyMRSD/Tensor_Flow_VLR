import tensorflow as tf
import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import datetime
import numpy as np
import os
import time

dir = os.path.dirname(os.path.realpath(__file__)) + 'logs/'
def load_image(path, size=112):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


# Load the VGG-16 model in the default graph
vgg_saver = tf.train.import_meta_graph('/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.meta')
# Access the graph
vgg_graph = tf.get_default_graph()

# Retrieve VGG inputs
img = load_image("cat.jpg")
img = img.reshape((1, 112, 112, 3))
input_vgg = tf.constant(np.ones([1, 112, 112, 3]), dtype=tf.float32)
#input_vgg = vgg_graph.get_tensor_by_name('Const:0')
print input_vgg.shape
# Choose which node you want to connect your own graph
#output_conv =vgg_graph.get_tensor_by_name('fc/xw_plus_b/MatMul:0')
output_conv =vgg_graph.get_tensor_by_name('avg_pool:0')
print output_conv
fc_weights =vgg_graph.get_tensor_by_name('fc/weights/read:0')
print "fc_weights",fc_weights
# output_conv =vgg_graph.get_tensor_by_name('conv2_2:0')
# output_conv =vgg_graph.get_tensor_by_name('conv3_3:0')
# output_conv =vgg_graph.get_tensor_by_name('conv4_3:0')
# output_conv =vgg_graph.get_tensor_by_name('conv5_3:0')
print "output shape-----",output_conv.shape
# Stop the gradient for fine-tuning
output_conv_sg = tf.stop_gradient(output_conv) # It's an identity function
print "output shape-----",output_conv_sg.shape
# Build further operations
weights = tf.get_variable('weights',
                            shape=[2048, 1000],
                            initializer=tf.random_normal_initializer(stddev=1e-1))
biases = tf.get_variable('biases',
                           shape=[1000],
                           initializer=tf.random_normal_initializer(stddev=1e-1))
a = tf.nn.xw_plus_b(output_conv, weights, biases)
#output_conv_shape = output_conv_sg.get_shape().as_list()
#W1 = tf.get_variable('W1', shape=[1, 1, output_conv_shape[3], 32], initializer=tf.random_normal_initializer(stddev=1e-1))
#b1 = tf.get_variable('b1', shape=[32], initializer=tf.constant_initializer(0.1))
#z1 = tf.nn.conv2d(output_conv_sg, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
#a = tf.nn.relu(z1)

with tf.Session() as sess:
  # Init v and v2   
  sess.run(tf.global_variables_initializer())
  # Now v1 holds the value 1.0 and v2 holds the value 2.0
  # We can now save all those values

  vgg_saver.save(sess, 'new_data-all.chkp')

train_writer = tf.summary.FileWriter(dir,
                                      sess.graph)