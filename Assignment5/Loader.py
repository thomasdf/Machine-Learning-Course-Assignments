import numpy as np
import os
from PIL import Image
from functools import reduce
from random import shuffle


base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
chars = "abcdefghijklmnopqrstuvwxyz"

def __loadArr():
	allimages = []
	chararr = []
	n_classes = len(chars)
	one_hot = []
	for index in range(n_classes):
		label = np.zeros([n_classes])
		label[index] = 1
		one_hot.append(label)

	for char in chars:
		for imgpath in os.listdir(assignment5dir + "/datasets/chars/" + char ):
			im = Image.open(assignment5dir + "/datasets/chars/" + char + "/" + imgpath)
			chararr.append((np.array(im), one_hot[chars.index(char)]))
		allimages.append(chararr)
		chararr = []
	return allimages

def add_labels(alldata):
	return __add_onehot_labels(alldata)

def __add_onehot_labels(alldata):
	n_classes = len(alldata)
	one_hot = []
	for index in range(n_classes):
		label = np.zeros([n_classes])
		label[index] = 1
		one_hot.append(label)

	newall = []
	for i in range(n_classes):
		chardata = []
		for image in alldata[i]:
			chardata.append((image, one_hot[i]))
		newall.append(chardata)

	return newall


def __split_data_add_duplicates(alldata, testsetfrac):
	"""Skewed dataset strategy: The label with most examples is used as a baseline. For all other labels, examples are duplicated to make the same amount of examples for each label"""
	num_images = sum([len(x) for x in alldata])
	char_most_images = max([len(x) for x in alldata])
	num_test_images_per_char = int(num_images*testsetfrac/len(alldata))
	test = []
	train = []

	for char in alldata:
		test.append(char[:num_test_images_per_char])
		expand = []
		for _ in range(int(np.ceil(char_most_images/len(char[num_test_images_per_char:])))):
			expand.append(char[num_test_images_per_char:])
		expand = flatmap(expand)
		train.append(expand[:char_most_images])

	return train, test

def __split_data_remove_data(alldata, testsetfrac):
	"""Skewed dataset strategy: The label where there exists fewest samples is used as a baseline. For all other labels, examples will be ignored."""
	num_images = sum([len(x) for x in alldata])
	char_fewest_images = min([len(x) for x in alldata])
	num_test_images_per_char = int(num_images * testsetfrac / len(alldata))
	test = []
	train = []

	for char in alldata:
		test.append(char[:num_test_images_per_char])
		train.append(char[num_test_images_per_char:char_fewest_images])

	return train, test

def __split_data(alldata, testsetfrac):
	"""Skewed dataset strategy: Original number of examples per label is kept."""
	num_images = sum([len(x) for x in alldata])
	num_test_images_per_char = int(num_images * testsetfrac / len(alldata))
	test = []
	train = []

	for char in alldata:
		test.append(char[:num_test_images_per_char])
		train.append(char[num_test_images_per_char:])

	return train, test

def shuffle_all_data(alldata):
	"""shuffles all images, to randomize training and test sets"""
	for char in alldata:
		shuffle(char)
	return alldata

def flatmap(set):
	return reduce(list.__add__, set)

def split_data(alldata, testsetfrac, splitter: callable = __split_data):
	return splitter(alldata, testsetfrac)

def __load(testsetfraq):
	alldata = __loadArr()
	shuffle_all_data(alldata)
	trainset, testset = split_data(alldata, testsetfraq, splitter=__split_data)
	trainset = flatmap(trainset)
	testset = flatmap(testset)

	shuffle(trainset)
	shuffle(testset)

	return trainset, testset

def load(testsetfraq):
	return __load(testsetfraq)
