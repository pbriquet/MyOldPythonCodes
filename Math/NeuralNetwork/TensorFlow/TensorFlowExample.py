import tensorflow as tf

def neural_network(x,W_0, W_1, W_2, b_0, b_1, b_2):
    h = tf.tanh(tf.matmul(x,W_0) + b_0)
    h = tf.tanh(tf.matmul(x,W_0) + b_0)
    h = tf.tanh(tf.matmul(x,W_0) + b_0)
    return tf.reshape(h, [-1])

