import numpy as np
import math
import matplotlib.pyplot as plt


def readCSV(path):
    return np.genfromtxt(path, delimiter=',')

def addOne(matrix):
    x = 1
    return np.hstack(([[x]] * len(matrix), matrix))

def getXandY(matrix):
    ylen, xlen = matrix.shape
    y = matrix[:,xlen-1]
    X = matrix[:,range(0,xlen-1)]

    return X, y

def sigma():
    a = 1

def train(X, y, w, steps, learningrate):
    wprev = w
    wnext = []
    for i in range(0, steps):
        sum = 0
        rowindex = 0
        for x in X:
            sum += (sigma(wprev, x) - y[rowindex])*x
        wnext = wprev - (learningrate * sum)
        wprev = wnext
        rowindex += 1
    return wnext

def sigma(w, x):
    #print w
    #print x
    z = x.dot(np.transpose(w))
    result = 1/(1+ math.exp(-z))
    #print result
    return result

def classify(X, w):
    ylen, xlen = X.shape
    #print ylen
    y = np.zeros(ylen)
    result = np.zeros(ylen)
    rowindex = 0
    for x in X:
        sigmares = sigma(w, x)
        result[rowindex] = sigmares
        res = round(sigmares)
        #print res
        y[rowindex] = res
        rowindex += 1
    return y, result

matrix = readCSV('datasets/classification/cl-train-2.csv')
matrix = addOne(matrix)
#print matrix
X, y = getXandY(matrix)
w = [0.00001, 0.00001, 0.00001]
steps = 1
learningrate = 0.0000001
w = train(X, y, w, steps, learningrate)

matrix = readCSV('datasets/classification/cl-train-2.csv')
matrix = addOne(matrix)
X, y = getXandY(matrix)

prediction, res = classify(X, w)
resultmatrix = np.column_stack((y, prediction, res))
correct = 0
for i in range(0, len(prediction)):
    if(prediction[i] == y[i]):
        correct += 1

print resultmatrix
print correct