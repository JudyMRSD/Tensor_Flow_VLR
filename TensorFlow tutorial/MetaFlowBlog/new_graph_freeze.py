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
    print num_units_in
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
    x = tf.nn.xw_plus_b(x, weights, biases)
    print "------4------"
    print weights 
    print biases 
    return x


# Load the VGG-16 model in the default graph
vgg_saver = tf.train.import_meta_graph('/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.meta')
# Access the graph
vgg_graph = tf.get_default_graph()

# Retrieve VGG inputs
img = load_image("cat.jpg")
img = img.reshape((1, 112, 112, 3))
#input_vgg = tf.constant(np.ones([1, 112, 112, 3]), dtype=tf.float32)
#input_vgg = vgg_graph.get_tensor_by_name('Const:0')
#print input_vgg.shape

output_conv =vgg_graph.get_tensor_by_name('avg_pool:0')
# Stop the gradient for fine-tuning
output_conv_sg = tf.stop_gradient(output_conv) # It's an identity function
print "output shape-----",output_conv_sg.shape
# Build further operations
#fc layer 

print "-----start fc-----"

print fc(output_conv_sg, 512)
print "-----------finish fc-------"

with tf.Session() as sess:
  # Init v and v2   
  sess.run(tf.global_variables_initializer())
  # Now v1 holds the value 1.0 and v2 holds the value 2.0
  # We can now save all those values

  vgg_saver.save(sess, 'new_data-all.chkp')

train_writer = tf.summary.FileWriter(dir,
                                      sess.graph)