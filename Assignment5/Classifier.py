from Assignment5 import CNN, KNN

def train(trainingdata, testingdata, classifier = CNN):
	return classifier.train(200, trainingdata, testingdata)

def run(data, classifier: callable = CNN, model = None):
	return classifier.run(data, model)
