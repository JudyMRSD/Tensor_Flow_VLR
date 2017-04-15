import tensorflow as tf
import numpy as np
t1 = np.ones([1, 112, 112, 3])
print t1.shape
t2 = np.ones([1, 112, 112, 3])
print t2.shape
t3 = np.concatenate([t1, t2], 3) 
#==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
t4 = np.concatenate([t1, t2], 1) 
#==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
print t3.shape
print t4.shape
# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
#print tf.shape(tf.concat([t3, t4], 0)) 
#==> [4, 3]
#print tf.shape(tf.concat([t3, t4], 1)) 
#==> [2, 6]