import os

import numpy as np

from Assignment5 import Loader


def preprocessArr(arr, flatten = True):
	if flatten:
		# return __flatten(normalizeImg(BWimg(arr)))
		return __flatten(normalizeImg(arr))

	#return normalizeImg(BWimg(arr))
	return normalizeImg(arr)
	#return __flatten(arr)

def __flatten(arr):
	return np.ravel(arr)

def BWimg(arr):
	arr[arr<255/2] = 0
	arr[arr>255/2] = 255
	return arr

def normalizeImg(arr):
	return arr * (1/255)

def denormalizeIImg(arr):
	return arr * 255

def preprocessData(set, preprocess: callable = preprocessArr):
	for i in range(len(set)):
		set[i] = (preprocess(set[i][0]), set[i][1])
	return set
