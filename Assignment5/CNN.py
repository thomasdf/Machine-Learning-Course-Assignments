import tensorflow as tf
import numpy as np

size = 20
n_classes = 26
__trainingdata = []
__labels = []
__testingdata = []



def model(x):
	#flatinput = tf.reshape(x, shape=[-1, int(size*size)])
	conv1 = tf.layers.conv2d(
		inputs=x,
		filters=32,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu,
	)

	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=2,
		strides=2,
	)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu,
	)

	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=2,
		strides=2,
	)

	pool2flat = tf.reshape(pool2, shape=[-1, int( (size/4) * (size/4))])

	fc = tf.layers.dense(
		inputs=pool2flat,
		units=1024,
		activation=tf.nn.relu,
	)

	logits = tf.layers.dense(
		inputs=fc,
		units=n_classes,
	)

	return logits

def __train(epochs):
	return 1337


def __createlabels(testingdata):
	one_hot = []
	for index in range(len(testingdata)):
		label = np.zeros([len(testingdata)])
		label[index] = 1
		one_hot.append(label)
	return one_hot


def train(trainingdata, testingdata, labels):
	__labels = __createlabels(testingdata)
	__testingdata = testingdata
	__trainingdata = trainingdata
	__train(10)
