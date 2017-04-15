#from convert import print_prob, load_image, checkpoint_fn, meta_fn

import os
os.environ["GLOG_minloglevel"] = "2"
import sys
import re
import numpy as np
import tensorflow as tf
import skimage.io
from synset import *

import resnet
import tensorflow as tf
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print "Top1: ", top1
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1

layers = 50

img = load_image("data/cat.jpg")
sess = tf.Session()

#new_saver = tf.train.import_meta_graph(meta_fn(layers))
path_model = '/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.meta'

path_model2 = '/Users/jinzhu/Google Drive/2017 spring/visual learning recognition/Project/Tensor Flow/tensorflow-resnet-MIT/data/tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt'
saver = tf.train.import_meta_graph(path_model)
print "9---------------"
saver.restore(sess, path_model2)
graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
print "prob_tensor"
print prob_tensor
images = graph.get_tensor_by_name("images:0")
print images
for op in graph.get_operations():
    print op.name

#init = tf.initialize_all_variables()
#sess.run(init)
print "graph restored"

batch = img.reshape((1, 224, 224, 3))
print "---------1------------"
feed_dict = {images: batch}
print "---------2------------"

prob = sess.run(prob_tensor, feed_dict=feed_dict)
print "---------3------------"

print_prob(prob[0])


