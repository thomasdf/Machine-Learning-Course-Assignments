from Assignment5 import CNN

def train(trainingdata, testingdata, classifier = CNN):
	CNN.train(500, trainingdata, testingdata)

def run(data, classifier: callable = CNN):
	return classifier.run(data)
