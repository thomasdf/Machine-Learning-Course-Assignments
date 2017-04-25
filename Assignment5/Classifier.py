from Assignment5 import CNN

def train(trainingdata, testingdata, classifier = CNN):
	CNN.train(1000, trainingdata, testingdata)

def run(data, classifier: callable = CNN):
	return classifier.run(data)
