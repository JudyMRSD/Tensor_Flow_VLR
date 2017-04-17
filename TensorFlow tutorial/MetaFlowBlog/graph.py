import tensorflow as tf
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
d = sess.run(c)
e = [1 ,2 ]
#print e.type
e = tf.convert_to_tensor(e)
print tf.cast(e, tf.float32)
