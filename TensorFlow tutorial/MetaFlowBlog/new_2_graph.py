#multiple graphs
#https://www.quora.com/How-does-one-create-separate-graphs-in-TensorFlow
#http://stackoverflow.com/questions/35955144/working-with-multiple-graphs-in-tensorflow
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

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


dir = os.path.dirname(os.path.realpath(__file__)) + 'logs/'
def load_image(path, size=112):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)

def fc(x, num_units_out=512):
    
    num_units_in = x.get_shape()[1]
    print "num in ", num_units_in
    print "-------0-----------"
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)
    print "------1------"
    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    print "------2------"
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    print "------3------"
    print weights 
    print biases 
    print x
    x = tf.nn.xw_plus_b(x, weights, biases)

    print "------4------"
    print weights 
    print biases 
    return x

def run_graph(train_dir='2graph_logs'):

  path_model = '/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.meta'

  path_model2 = '/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt'
  
  init = tf.global_variables_initializer()
  
  sess = tf.Session()
  sess.run(init)

  saver = tf.train.import_meta_graph(path_model)
  print "9---------------"
  saver.restore(sess, path_model2)
  graph = tf.get_default_graph()

  img = load_image("cat.jpg")
  img = img.reshape((1, 112, 112, 3))

  img2 = load_image("cat.jpg")
  img2 = img2.reshape((1, 112, 112, 3))

  
  with tf.variable_scope("resnet1"):
    output_conv =graph.get_tensor_by_name('avg_pool:0')
    images = graph.get_tensor_by_name("images:0")
    output_conv_sg = tf.stop_gradient(output_conv) # It's an identity function
    
  with tf.variable_scope("resnet2"):
    output_conv2=graph.get_tensor_by_name('avg_pool:0')
    images2 = graph.get_tensor_by_name("images:0")
    output_conv_sg2 = tf.stop_gradient(output_conv) # It's an identity function

  print "-----------10---------"
  #print "output_conv", output_conv #Tensor("avg_pool:0", shape=(?, 2048), dtype=float32, device=/device:CPU:0)
  #print "output_conv_sg",output_conv_sg #Tensor("StopGradient:0", shape=(?, 2048), dtype=float32)


  print "graph restored"

  print "-----------11-----------"
  output_1 = sess.run([output_conv_sg], feed_dict={images:img})
  output_2 = sess.run([output_conv_sg2], feed_dict={images:img2})

  print "output1"
  #convert to tensorflow output_1

  #reshape(t, [2, -1])


  output_1_tf = tf.stack(output_1)
  output_2_tf = tf.stack(output_2)
  print "------reshape-"
  print output_1_tf
  print output_2_tf
  reshape_output_1_tf = tf.reshape(output_1_tf,[1,-1])
  reshape_output_2_tf = tf.reshape(output_2_tf,[1,-1])
  #tf_2 = tf.squeeze(output_1_tf)
  with tf.variable_scope("resnet1"):
    fc(reshape_output_1_tf, 512)
  with tf.variable_scope("resnet2"):
    fc(reshape_output_2_tf, 512)
  #output_2 = sess.run(fc_output)
  print "----------------12------------"
  print graph
  print "-------------13--------------"

  train_writer = tf.summary.FileWriter(train_dir,
                                      sess.graph)

run_graph()