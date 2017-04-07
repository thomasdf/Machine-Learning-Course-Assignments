#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tf_wgan_tdt4173.py: Rudimentary implementation of a wasserstein GAN that
# is to be used as a base for the second programming task in assignment 4.
#
import sys
import os
from tensorflow.python.client import device_lib
import numpy as np
import tensorflow as tf

from Assignment4.extramaterial import helpers

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(device_lib.list_local_devices())

# You might want to alter the learning rate, number of epochs, and batch size
lr = 0.0005
nb_epochs = 50000
batch_size = 64

# Set to `None` if you do not want to write out images
path_to_images = './generated_images_wgan3'

z_size = 10
x_size = 28*28


# Defined at the top because we need it for initialising weights
def create_weights(shape):
    # See paper by Xavier Glorot and Yoshua Bengio for more information:
    # "Understanding the difficulty of training deep feedforward neural networks"
    # We employ the Caffe version of the initialiser: 1/(in degree)
    return tf.random_normal(shape, stddev=1/shape[0])

def create_biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#
# Creation of generator and discriminator networks START here
# Task (a) is to improve the generator and discriminator networks as they
# currently do not do very much
#

# Define weight matrices for the generator
# Note: Input of the first layer *must* be `z_size` and the output of the
# *last* layer must be `x_size`
num_neurons_hl_1_generator = 98
num_neurons_hl_2_generator = 392
num_neurons_hl_3_generator = 1568
weights_G = {
        'w1': tf.Variable(create_weights((z_size, num_neurons_hl_1_generator))),
        'b1': tf.Variable(create_biases([num_neurons_hl_1_generator])),
        'w2': tf.Variable(create_weights((num_neurons_hl_1_generator, num_neurons_hl_2_generator))),
        'b2': tf.Variable(create_biases([num_neurons_hl_2_generator])),
        'w3': tf.Variable(create_weights((num_neurons_hl_2_generator, num_neurons_hl_3_generator))),
        'b3': tf.Variable(create_biases([num_neurons_hl_3_generator])),
        'w4': tf.Variable(create_weights((num_neurons_hl_3_generator, x_size))),
        'b4': tf.Variable(create_biases([x_size]))
}

def generator(z, weights):
    z1 = tf.add(tf.matmul(z, weights['w1']), weights['b1'])
    h1 = tf.nn.relu(z1)
    h2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
    h2 = tf.nn.relu(h2)
    h3 = tf.add(tf.matmul(h2, weights['w3']), weights['b3'])
    h3 = tf.nn.relu(h3)
    z2 = tf.add(tf.matmul(h3, weights['w4']), weights['b4'])
    out = tf.nn.relu(z2)

    # Return model and weight matrices
    return out



# Define weight matrices for the discriminator
# Note: Input will always be `x_size` and output will always be 1
num_neurons_hl_1_discriminator = 147
num_neurons_hl_2_discriminator = 74
num_neurons_hl_3_discriminator = 37
weights_D = {
    'w1': tf.Variable(create_weights((x_size, num_neurons_hl_1_discriminator))),
    'b1': tf.Variable(create_biases([num_neurons_hl_1_discriminator])),
    'w2': tf.Variable(create_weights((num_neurons_hl_1_discriminator, num_neurons_hl_2_discriminator))),
    'b2': tf.Variable(create_biases([num_neurons_hl_2_discriminator])),
    'w3': tf.Variable(create_weights((num_neurons_hl_2_discriminator, num_neurons_hl_3_discriminator))),
    'b3': tf.Variable(create_biases([num_neurons_hl_3_discriminator])),
    'w4': tf.Variable(create_weights((num_neurons_hl_3_discriminator, 1))),
    'b4': tf.Variable(create_biases([1]))
}

def discriminator(x, weights):
    z1 = tf.add(tf.matmul(x, weights['w1']), weights['b1'])
    h1 = tf.nn.relu(z1)
    h2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
    h2 = tf.nn.relu(h2)
    h3 = tf.add(tf.matmul(h2, weights['w3']), weights['b3'])
    h3 = tf.nn.relu(h3)
    z2 = tf.add(tf.matmul(h3, weights['w4']), weights['b4'])
    #out = tf.nn.relu(z2)
    out = z2

    # Return model and weight matrices
    return out

#
# Creation of generator and discriminator networks END here
#

# Weight clipping (default `c` from the WGAN paper)
c = 0.01
clipped_D = [w.assign(tf.clip_by_value(w, -c, c)) for w in weights_D.values()]

# Definition of how Z samples are generated
z_sampler = lambda nb, dim: np.random.uniform(-1.0, 1.0, size=(nb, dim))

# Load MNIST
mnist = helpers.load_mnist_tf('./mnist')

# Define model entry-points (Z - generator, X - discriminator)
Z = tf.placeholder(tf.float32, shape=(None, z_size))
X = tf.placeholder(tf.float32, shape=(None, x_size))

# Define the different components of a GAN
sample = generator(Z, weights_G)
fake_hat = discriminator(sample, weights_D)
real_hat = discriminator(X, weights_D)

# Define error functions
error_G = - tf.reduce_mean(fake_hat)
error_D = tf.reduce_mean(real_hat) - tf.reduce_mean(fake_hat)

# Specify that we will use RMSProp (one optimiser for each model)
optimiser_G = tf.train.RMSPropOptimizer(lr).minimize(error_G,
    var_list=weights_G.values())
optimiser_D = tf.train.RMSPropOptimizer(lr).minimize(-error_D,
    var_list=weights_D.values())

# Generate Op that initialises global variables in the graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialise variables and start the session
    sess.run(init)

    if path_to_images:
        helpers.create_dir(path_to_images)

    # Run a set number of epochs (default `n_critic` from the WGAN paper)
    #lr = 0.00000001
    n_critic = 5
    plot_error_gen = []
    plot_error_disc = []
    plot_epoch = []
    for epoch in range(nb_epochs):
        #if epoch == 5:
            #lr = 0.0005
        for _ in range(n_critic):

            # Retrieve a batch from MNIST
            X_batch, _ = mnist.train.next_batch(batch_size)

            # Clip weights and run one step of the optimiser for D
            sess.run(clipped_D)
            sess.run(optimiser_D, feed_dict={Z: z_sampler(batch_size, z_size),
                                             X: X_batch})

        # Run one step of the optimiser for G
        sess.run(optimiser_G, feed_dict={Z: z_sampler(batch_size, z_size)})

        # Print out some information every nth iteration
        if epoch % 100 == 0:
            err_G = sess.run(error_G, feed_dict={Z: z_sampler(batch_size, z_size)})
            err_D = sess.run(error_D, feed_dict={Z: z_sampler(batch_size, z_size),
                                                 X: X_batch})

            print('Epoch: ', epoch)
            print('\t Generator error:\t {:.4f}'.format(err_G))
            print('\t Discriminator error:\t {:.4f}'.format(err_D))
            plot_epoch.append(epoch)
            plot_error_gen.append(err_G)
            plot_error_disc.append(err_D)

        # Plot the image generated from 64 different samples to a directory
        if path_to_images and epoch % 1000 == 0:
            samples = sess.run(sample, feed_dict={Z: z_sampler(64, z_size)})

            figure = helpers.plot_samples(samples)
            plt.savefig('{}/{}.png'.format(path_to_images, str(epoch)),
                        bbox_inches='tight')
            plt.close()
    plt.figure()
    gen, = plt.plot(plot_epoch, plot_error_gen, label='gen error')
    disc, = plt.plot(plot_epoch, plot_error_disc, label='disc error')
    plt.legend()
    #plt.axis([ 0, nb_epochs, 0, 1.1 ])
    plt.show()

del sess
sys.exit(0)
