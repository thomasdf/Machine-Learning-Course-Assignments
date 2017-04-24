import os

def preprocessData(arr):
	normalized = normalizeImg(arr)


def normalizeImg(arr):
	return arr * (1 / 255)

def denormalizeIImg(arr):
	return arr * 255

def preprocessAllData(alldata, preprocess: callable = preprocessData):
	for char in alldata:
		for img in char:
			img = preprocess(img)
	return alldata



