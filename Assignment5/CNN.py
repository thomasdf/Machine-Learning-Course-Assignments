import tensorflow as tf
import numpy as np
import os
from random import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
size = 20
log_modulo = 10
n_classes = 26

x = tf.placeholder("float", [None, int(size*size)])
y = tf.placeholder("float", [None, n_classes])

def model(x, isTraining = True):
	input = tf.reshape(x, shape=[-1, size, size, 1], name="input-reshape")
	conv1 = tf.layers.conv2d(
		inputs=input,
		filters=32,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu,
		name="conv1"
	)

	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=2,
		strides=2,
		name="pool1"
	)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu,
		name="conv2"
	)

	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=2,
		strides=2,
		name="pool2"
	)

	pool2flat = tf.reshape(pool2, shape=[-1, int( (size/4) * (size/4)) * 64], name="pool2flatten")

	fc1 = tf.layers.dense(
		inputs=pool2flat,
		units=1024,
		activation=tf.nn.relu,
		name="fc1"
	)

	droput = tf.layers.dropout(
		inputs=fc,
		rate=0.4,
		training=isTraining,
	)

	logits = tf.layers.dense(
		inputs=droput,
		units=n_classes,
		name="logits"
	)

	return logits

def accuracy(testingdata, testinglabels, prediction):
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	acc_func = tf.reduce_mean(tf.cast(correct, "float"))
	acc = acc_func.eval({x: testingdata, y: testinglabels})
	return acc

def __train(epochs, trainingset, testingset, lr = 1e-3):
	testingdata, testinglabels = __splitLabels(testingset)
	logits = model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver()
		for epoch in range(epochs):
			data, labels = augmentTrainingSet(trainingset)
			_, c = sess.run([optimizer, cost], feed_dict={x: data, y: labels})

			if epochs % log_modulo == 0:
				acc = accuracy(testingdata, testinglabels, logits)
				print("Epoch", epoch, "\tLoss", c, "\tAccuracy", acc)

		saver.save(sess, assignment5dir + "/savedmodels/" + "model.chkpt")


def __run(data, labels):
	return 1337

def augmentTrainingSet(trainingset):
	return __shuffleTrainingSet(trainingset)

def __splitLabels(set):
	labels = [pair[1] for pair in set]
	data = [pair[0] for pair in set]

	return data, labels

def __shuffleTrainingSet(trainingdata):
	shuffle(trainingdata)
	return __splitLabels(trainingdata)


def splitLabels(trainingdata, testingdata):
	trainingdata, traininglabels = __splitLabels(trainingdata)
	testingdata, testinglabels = __splitLabels(testingdata)
	return trainingdata, traininglabels, testingdata, testinglabels


def train(epochs, trainingset, testingset):
	__train(epochs, trainingset, testingset)
