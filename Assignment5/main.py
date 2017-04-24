import os
from Assignment5 import Loader, Preprocess, Classifier

base_dir = os.path.dirname(os.path.dirname(__file__))
assignment5dir = base_dir + "/Assignment5"
chars = "abcdefghijklmnopqrstuvwxyz"

def main():
	alldata = Loader.load()
	preprocessed = Preprocess.preprocessAllData(alldata)
	trainingdata, testingdata = Loader.split_data(preprocessed, 0.2)
	train_classifier(trainingdata, testingdata)

def train_classifier(trainingdata, testingdata):
	Classifier.train(trainingdata, testingdata)


def run_classifier(data):
	Classifier.run(data)

def preprocess(alldata):
	Preprocess.preprocessAllData(alldata)


main()
