import tensorflow as tf
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
size = 20
log_modulo = 10
n_classes = 26

def __train(epochs, trainingset, testingset, lr=1e-3):
	testingdata, testinglabels = __splitLabels(testingset)
	data, labels = augmentTrainingSet(trainingset)

	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(data, labels)
	pred = neigh.predict(testingdata)

	accuracy = np.sum(pred == testinglabels).astype(float) / len(testinglabels)
	print("Accuracy:", accuracy, "%")

	return neigh

def __run(data, model):
	neigh = model
	return neigh.predict(data)



def augmentTrainingSet(trainingset):
	# return __shuffleTrainingSet(trainingset)
	return __splitLabels(trainingset)


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
	return __train(epochs, trainingset, testingset)


def run(data, model):
	return __run(data, model)
