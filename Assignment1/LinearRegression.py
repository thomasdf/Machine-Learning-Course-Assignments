import numpy as np
import csv
from matplotlib.pylab import scatter, show


def readCSV(path):
    return np.genfromtxt(path, delimiter=',')

def addOne(array):
    x = 1
    return np.hstack(([[x]] * len(array), array))

def linearRegression(array):
    #print(array)
    print(len(array[1][:]))
    X = [[]]
    for i in range(len(array)):

    X = array[0:len(array[0])-1]
    print (len(X))
    y = array[:][1]
    print(X)
    #print(y)
    # Xt = np.matrix(X)
    # Xt = Xt.T
    # XtX = np.dot(X, Xt)
    # XtXinverse = np.linalg.inv(XtX)
    # w = np.dot(np.dot(XtXinverse, Xt), y)
    # print(w)

#array = addOne(readCSV('datasets/regression/reg-1d-test.csv'))
array = readCSV('datasets/regression/reg-2d-train.csv')
linearRegression(array)
#scatter(array[:,0], array[:,1])
#show()

