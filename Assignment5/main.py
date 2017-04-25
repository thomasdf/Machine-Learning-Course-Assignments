import os
from Assignment5 import Loader, Preprocess, Classifier

base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
chars = "abcdefghijklmnopqrstuvwxyz"

def main():
	trainingdata, testingdata = Loader.load(0.05)
	trainingdata = preprocess(trainingdata)
	testingdata = preprocess(testingdata)
	train_classifier(trainingdata, testingdata)

def train_classifier(trainingdata, testingdata):
	Classifier.train(trainingdata, testingdata)

def run_classifier(data):
	Classifier.run(data)

def preprocess(set):
	return Preprocess.preprocessData(set)



main()
