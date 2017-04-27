import os

from PIL import Image
import numpy as np

from Assignment5 import Loader, Preprocess, Classifier, SlidingWindow, KNN, CNN

base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
chars = "abcdefghijklmnopqrstuvwxyz"

def main():
	#train_classifier(CNN)
	classifyRandImg("q")
	#slidingwindowclassify(2)

	#knnclassifier = train_classifier(KNN)
	#slidingwindowclassify(2, KNN, knnclassifier)


def train_classifier(classifier):
	trainingdata, testingdata = Loader.load(0.2)
	trainingdata = preprocess(trainingdata)
	testingdata = preprocess(testingdata)
	return Classifier.train(trainingdata, testingdata, classifier=classifier)

def run_classifier(data, classifier = CNN, model = None):
	return Classifier.run(data, classifier, model)

def classifyRandImg(char):
	im = Loader.loadRandomCharImg(char)
	previewim = im.copy()
	preprocessed = preprocess(im, Preprocess.preprocessArr)
	flattened = np.ravel(preprocessed)
	reshaped = np.reshape(flattened, [1, 400])
	probabilities = run_classifier(reshaped)
	classification = np.argmax(probabilities)

	print("Probabilities:", probabilities)
	print("Assigned class:", classification)
	np.reshape(im, [20,20])
	Image.fromarray(previewim).show()

def slidingwindowclassify(img_num, classifier = CNN, KNNclassifier = None):
	img, imgarr = Loader.loadSlidingWindowClassifierImg(img_num)
	#img = img.crop((0,14,img.size[0], img.size[1]))
	#imgarr = np.array(img)

	img = SlidingWindow.sliding_classify(img, arr=imgarr, size=20, stride=20, model=classifier, treshold=0.2, scaled_shade=False, shader=SlidingWindow.shade2dgrayscale, trained = KNNclassifier)
	img.show()


def preprocess(data, preprocess = Preprocess.preprocessData):
	return preprocess(data)



main()
