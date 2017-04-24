from Assignment5 import CNN

def train(trainingdata, testingdata, classifier: callable = CNN):
	classifier.train(trainingdata, testingdata)

def run(data, classifier: callable = CNN):
	return classifier.run(data)
