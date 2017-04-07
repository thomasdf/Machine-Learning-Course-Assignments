#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tf_xor_tdt4173.py: TensorFlow example for learning the XOR function.
#
import sys
import os
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Assignment4.extramaterial import helpers

os.environ["CUDA_VISIBLE_DEVICES"]="2"
print(device_lib.list_local_devices())

X_train, y_train, X_test, y_test = helpers.load_task1_data("/extramaterial/data/cl-train.csv", "/extramaterial/data/cl-test.csv")
# Define dataset
#DATA_X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
#DATA_y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define model entry-points
X = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))

num_inputs = 2
num_nodes_hl_1 = 40
num_nodes_hl_2 = 40
num_outputs = 1


# Create weights and gather them in a dictionary
weights = {'w1': tf.Variable(tf.random_uniform([num_inputs, num_nodes_hl_1], -1, 1)),
           'b1': tf.Variable(tf.zeros([num_nodes_hl_1])),
           'w2': tf.Variable(tf.random_uniform([num_nodes_hl_1, num_nodes_hl_2], -1, 1)),
           'b2': tf.Variable(tf.zeros([num_nodes_hl_2])),
           'w3': tf.Variable(tf.random_uniform([num_nodes_hl_2, num_outputs], -1, 1)),
           'b3': tf.Variable(tf.zeros([num_outputs]))}

# Define model as a computational graph
z1 = tf.add(tf.matmul(X, weights['w1']), weights['b1'])
h1 = tf.sigmoid(z1)
z2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
h2 = tf.sigmoid(z2)
z3 = tf.add(tf.matmul(h2, weights['w3']), weights['b3'])
y_hat = tf.sigmoid(z3)

# Define error functions
error = - tf.reduce_mean(tf.multiply(y, tf.log(y_hat)) +
                         tf.multiply(1 - y, tf.log(1 - y_hat)))

# Specify which optimiser to use (`lr` is the learning rate)
lr = 3.0
optimiser = tf.train.GradientDescentOptimizer(lr).minimize(
    error, var_list=weights.values())

# Generate Op that initialises global variables in the graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialise variables and start the session
    sess.run(init)

    # Run a set number of epochs
    nb_epochs = 1000
    plot_error_test = []
    plot_error_train = []
    plot_epoch = []
    for epoch in range(nb_epochs):
        sess.run(optimiser, feed_dict={X: X_train, y: y_train})

        # Print out some information every nth iteration
        err_test = sess.run(error, feed_dict={X: X_test, y: y_test})
        plot_error_test.append(err_test)
        err_train = sess.run(error, feed_dict={X: X_train, y: y_train})
        plot_error_train.append(err_train)
        plot_epoch.append(epoch)
        if epoch % 100 == 0:
            print('Epoch: ', epoch, '\t Test Error: ', err_test, '\t', '\t Train Error: ', err_train, '\t')

    # Print out the final predictions
    err_test = sess.run(error, feed_dict={X: X_test, y: y_test})
    err_train = sess.run(error, feed_dict={X: X_train, y: y_train})
    print('Epoch: ', epoch, '\t Test Error: ', err_test, '\t', '\t Train Error: ', err_train, '\t')

    plt.figure()
    plt.plot(plot_epoch, plot_error_train, label='training error')
    plt.plot(plot_epoch, plot_error_test, label='test error')
    plt.axis([ 0, nb_epochs, 0, 1.1 ])
    plt.show()
    #helpers.plot_curves('Epoch', plot_epoch, 'Train', plot_error_train, 'Test', plot_error_test)

del sess
sys.exit(0)
