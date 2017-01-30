import numpy as np
import csv
from matplotlib.pylab import scatter, show


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

def linearRegression(X, y):
    Xt = X.transpose()
    w = np.linalg.inv(Xt.dot(X)).dot(Xt).dot(y)
    print w

#array = addOne(readCSV('datasets/regression/reg-1d-test.csv'))
matrix = readCSV('datasets/regression/reg-2d-train.csv')
matrix = addOne(matrix)
X, y = getXandY(matrix)
w = linearRegression(X, y)

#scatter(array[:,0], array[:,1])
#show()

