# Trying to define the simplest possible neural net where the output layer of the neural net is a single
# neuron with a "continuous" (a.k.a floating point) output.  I want the neural net to output a continuous
# value based off one or more continuous inputs.  My real problem is more complex, but this is the simplest
# representation of it for explaining my issue.  Even though I've oversimplified this to look like a simple
# linear regression problem (y=m*x), I want to apply this to more complex neural nets.  But if I can't get
# it working with this simple problem, then I won't get it working for anything more complex.

import tensorflow as tf
import random
import numpy as np

INPUT_DIMENSION = 1
OUTPUT_DIMENSION = 1
TRAINING_RUNS = 100
BATCH_SIZE = 10000
VERF_SIZE = 1


# Generate two arrays, the first array being the inputs that need trained on, and the second array containing outputs.
def generate_test_point():
    x = random.uniform(-8, 8)

    # To keep it simple, output is just -x.
    out = -x

    return (np.array([x]), np.array([out]))


# Generate a bunch of data points and then package them up in the array format needed by
# tensorflow
def generate_batch_data(num):
    xs = []
    ys = []

    for i in range(num):
        x, y = generate_test_point()

        xs.append(x)
        ys.append(y)

    return (np.array(xs), np.array(ys))


# Define a single-layer neural net.  Originally based off the tensorflow mnist for beginners tutorial

# Create a placeholder for our input variable
x = tf.placeholder(tf.float32, [None, INPUT_DIMENSION])

# Create variables for our neural net weights and bias
W = tf.Variable(tf.zeros([INPUT_DIMENSION, OUTPUT_DIMENSION]))
b = tf.Variable(tf.zeros([OUTPUT_DIMENSION]))

# Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
# tutorial, I have removed the softmax op.  My expectation is that 'net' will return a floating point
# value.
net = tf.matmul(x, W) + b

# Create a placeholder for the expected result during training
expected = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION])

# Same training as used in mnist example
loss = tf.reduce_mean(tf.square(expected - net))
# cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# Perform our training runs

for i in range(TRAINING_RUNS):
    print("trainin run: ", i, )

    batch_inputs, batch_outputs = generate_batch_data(BATCH_SIZE)

    # I've found that my weights and bias values are always zero after training, and I'm not sure why.
    sess.run(train_step, feed_dict={x: batch_inputs, expected: batch_outputs})

    # Test our accuracy as we train...  I am defining my accuracy as the error between what I
    # expected and the actual output of the neural net.
    # accuracy = tf.reduce_mean(tf.sub( expected, net))
    accuracy = tf.subtract(expected, net)  # using just subtract since I made my verification size 1 for debug
    # tf.subtract()
    # Uncomment this to debug
    # import pdb; pdb.set_trace()
    print("W=%f, b=%f" % (sess.run(W), sess.run(b)))

    batch_inputs, batch_outputs = generate_batch_data(VERF_SIZE)
    result = sess.run(accuracy, feed_dict={x: batch_inputs, expected: batch_outputs})
    value = sess.run(net, feed_dict={x: batch_inputs})
    print("    progress: ")
    print("      inputs: ", batch_inputs)
    print("      outputs:", batch_outputs)
    print("      value:", value)
    print("      actual: ", result)