import tensorflow as tf
import numpy as np
import os
from tensorflow.python.client import device_lib

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
print(device_lib.list_local_devices())

#Data
DATA_X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
DATA_Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

#TF-variables
X = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))

#initialize variables
weights = { 'w1': tf.Variable(tf.random_uniform([2,2], -1, 1)),
            'b1': tf.Variable(tf.zeros([2])),
            'w2': tf.Variable(tf.random_uniform([2,2], -1, 1)),
            'b2': tf.Variable(tf.zeros([1])),
           }

#notes: tf.add(): element-wise addition, tf.matmul(): matrix multiplication, tf.sigmoid(): logistic function

#implement neural network
z1 = tf.add(tf.matmul(X, weights['w1']), weights['b1'])
h1 = tf.sigmoid(z1)
z2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
y_hat = tf.sigmoid(z2)

#implement error-function. We use tf.reduce_mean, which reduces the size of the tensor dimensions, by taking the mean value
error = - tf.reduce_mean(tf.multiply(y, tf.log(y_hat)) + tf.multiply(1-y, tf.log(1 - y_hat)))

lr = 10.0
optimiser = tf.train.GradientDescentOptimizer(lr).minimize(error, var_list=weights.values())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    nb_epochs = 1000
    for epoch in range(nb_epochs):
        sess.run(optimiser, feed_dict={X: DATA_X, y: DATA_Y})
        if epoch % 100 == 0:
            print("Epoch: " + str(epoch))
