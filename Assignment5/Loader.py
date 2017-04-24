import numpy as np
import os
from PIL import Image


base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
chars = "abcdefghijklmnopqrstuvwxyz"

def __loadArr():
	allimages = []
	chararr = []
	for char in chars:
		for imgpath in os.listdir(assignment5dir + "/datasets/chars/" + char ):
			im = Image.open(assignment5dir + "/datasets/chars/" + char + "/" + imgpath)
			chararr.append(np.array(im))
		allimages.append(chararr)
		chararr = []
	return allimages

def __split_data(alldata, testsetfrac):
	num_images = sum([len(x) for x in alldata])
	num_images_per_char = int(num_images*testsetfrac/len(alldata))

	test = []
	train = []

	for char in alldata:
		test.append(char[:num_images_per_char])
		train.append(char[num_images_per_char:])

	return train, test


def split_data(alldata, testsetfrac, splitter: callable = __split_data):
	return splitter(alldata, testsetfrac)

def init():
	arrays = __loadArr()
	asnps = []
	for array in arrays:
		asnps.append(np.array(array))
	return np.array(asnps)

def load():
	return init()
