import numpy as np
import csv
from matplotlib.pylab import scatter, show
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

def linearRegression(X, y):
    Xt = X.transpose()
    w = np.linalg.inv(Xt.dot(X)).dot(Xt).dot(y)
    return w

def graph2d(formula, x_range, matrix, w):
    x = np.array(x_range)
    y = formula(x,w)
    plt.plot(x, y)
    plt.scatter(matrix[:,0], matrix[:,1])
    plt.show()

def graph3d(formula, x_range, matrix, w):
    x = np.array(x_range)
    y = formula(x,w)
    plt.plot(x, y)
    plt.scatter(matrix[:,0], matrix[:,1], matrix[:,2])
    plt.show()

def firstdegreeformula(x, w):
    return w[1]*x + w[0]

def seconddegreeformula(x, w):
    return w[2]*x + w[1] * x + w[0]

def meansquarederror(matrix, w):
    X, y = getXandY(matrix)
    mse = (np.linalg.norm((X.dot(w[range(1,len(w))])-y))**2)/len(y)
    print mse


#array = addOne(readCSV('datasets/regression/reg-1d-test.csv'))
matrix = readCSV('datasets/regression/reg-1d-train.csv')
matrix = addOne(matrix)
X, y = getXandY(matrix)
w = linearRegression(X, y)
matrix = readCSV('datasets/regression/reg-1d-test.csv')
graph2d(firstdegreeformula, np.arange(0, 1, 0.01), matrix, w)
meansquarederror(matrix, w)

matrix = readCSV('datasets/regression/reg-2d-train.csv')
matrix = addOne(matrix)
X, y = getXandY(matrix)
w = linearRegression(X, y)
matrix = readCSV('datasets/regression/reg-2d-test.csv')
graph3d(seconddegreeformula, np.arange(0, 1, 0.01), matrix, w)

meansquarederror(matrix, w)


