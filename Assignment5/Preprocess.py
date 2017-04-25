import os

import numpy as np

from Assignment5 import Loader


def __preprocessData(arr):
	return __flatten(normalizeImg(arr))
	#return __flatten(arr)

def __flatten(arr):
	return np.ravel(arr)

def normalizeImg(arr):
	return arr * (1 / 255)

def denormalizeIImg(arr):
	return arr * 255

def preprocessData(set, preprocess: callable = __preprocessData):
	for i in range(len(set)):
		set[i] = (preprocess(set[i][0]), set[i][1])
	return set
