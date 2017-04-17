import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

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


activation = tf.nn.relu


def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True):
    c = Config()
    c['bottleneck'] = bottleneck
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)
        scale1_x = x 

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)
        scale2_x = x 

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)
        scale3_x = x 

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)
        scale4_x = x 

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)
        scale5_x = x 

    # post-net
    x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")
    avg_pool_x = x
    #flatten 4x2048
    x = tf.reshape(x, [1,-1])
    flatten_x = x 
    print "shape of x ", x.shape

    fc_x = 0
    if num_classes != None:
        with tf.variable_scope('fc'):
            x = fc(x)
            fc_x = x 
    tf.summary.scalar('scale1_x', scale1_x)
    return x, scale1_x,scale2_x,scale3_x,scale4_x,scale5_x, avg_pool_x, flatten_x,fc_x  


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x,
                    is_training,
                    num_blocks=3, # 6n+2 total weight layers will be used.
                    use_bias=False, # defaults to using batch norm
                    num_classes=10):
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['num_classes'] = num_classes
    logits=inference_small_config(x, c)
    return logits 

def inference_small_config(x, c):
    c['bottleneck'] = False
    c['ksize'] = 3
    c['stride'] = 1
    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 16
        c['block_filters_internal'] = 16
        c['stack_stride'] = 1
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)
        x = stack(x, c)

    with tf.variable_scope('scale2'):
        c['block_filters_internal'] = 32
        c['stack_stride'] = 2
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['block_filters_internal'] = 64
        c['stack_stride'] = 2
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")

    if c['num_classes'] != None:
        with tf.variable_scope('fc'):
            x = fc(x)

    return x


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb * 255.0)
    bgr = tf.concat(axis=3, values=[blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)

    return loss_


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer())
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x

#def fc(x, c):
def fc(x):
    num_units_in = x.get_shape()[1]
    num_units_out = 512
    #num_units_out = c['fc_units_out']
    print ("fc layer in, out shape----------", num_units_in, num_units_out)
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


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


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]

    shape = [ksize, ksize, filters_in, filters_out]
    print ("conv layer shape: ksize, ksize, filters_in, filters_out", filters_in)
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def load_image(path, size=112):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


def test_graph(train_dir='resnet_new_fc_logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    #input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    #4 observation frames 
    img1 = load_image("data/cat.jpg")
    img2 = load_image("data/cat.jpg")
    img3 = load_image("data/cat.jpg")
    img4 = load_image("data/cat.jpg")
    img1 = img1.reshape((112, 112, 3))
    img2 = img2.reshape((112, 112, 3))
    img3 = img3.reshape((112, 112, 3))
    img4 = img4.reshape((112, 112, 3))

    img5 = load_image("data/cat.jpg")
    img5 = img5.reshape((1,112, 112, 3))
    with tf.variable_scope("resnet1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        input_tensor_o = tf.constant(np.ones([4, 112, 112, 3]), dtype=tf.float32)
        o, scale1_o,scale2_o,scale3_o,scale4_o,scale5_o, avg_pool_o, flatten_o,fc_o = inference(input_tensor_o, is_training=False, num_classes=1000)

    with tf.variable_scope("resnet2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        input_tensor_t = tf.constant(np.ones([1, 112, 112, 3]), dtype=tf.float32)
        t, scale1_t,scale2_t,scale3_t,scale4_t,scale5_t, avg_pool_t, flatten_t,fc_t  = inference(input_tensor_t, is_training=False, num_classes=1000)

    result = fc(o)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    o1,o2,o3,o4,o5,o6,o7,o8,o9 = sess.run([o, scale1_o,scale2_o,scale3_o,scale4_o,scale5_o, avg_pool_o, flatten_o, fc_o], feed_dict={input_tensor_o: [img1,img2,img3,img4]})
    t1,t2,t3,t4,t5,t6,t7,t8,t9 = sess.run([t, scale1_t,scale2_t,scale3_t,scale4_t,scale5_t, avg_pool_t, flatten_t, fc_t ], feed_dict={input_tensor_t: img5})
    saver = tf.train.Saver(tf.global_variables())

    print o1.shape,o2.shape,o3.shape,o4.shape,o5.shape,o6.shape,o7.shape,o8.shape,o9.shape
    print t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape,t7.shape,t8.shape,t9.shape
    #(1, 512) (4, 56, 56, 64) (4, 28, 28, 256) (4, 14, 14, 512) (4, 7, 7, 1024) (4, 4, 4, 2048) (4, 2048) (1, 8192) (1, 512)
    #(1, 512) (1, 56, 56, 64) (1, 28, 28, 256) (1, 14, 14, 512) (1, 7, 7, 1024) (1, 4, 4, 2048) (1, 2048) (1, 2048) (1, 512)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    
 
   
    #print o1.shape,o2.shape,o3.shape,o4.shape,o5.shape,o6.shape,o7.shape,o8.shape 
    #print t1.shape,t2.shape,t3.shape,t4.shape,t5.shape,t6.shape,t7.shape,t8.shape 

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=0)



#test_graph()