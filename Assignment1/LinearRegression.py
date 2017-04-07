import numpy as np
import csv
from matplotlib.pylab import scatter, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

def graph2d(formula, x_range, matrixTrain, matrixTest, w):
    x = np.array(x_range)
    y = formula(x,w)
    plt.plot(x, y)
    plt.scatter(matrixTrain[:,0], matrixTrain[:,1], marker='o', c='r')
    plt.scatter(matrixTest[:,0], matrixTest[:,1], marker='^', c='g')
    plt.show()

def graph3d(formula, x1_range, x2_range,  matrixTest,  matrixTrain, w):
    x1, x2 = np.meshgrid(x1_range, x2_range)
    z = w[2]*x2 + w[1] * x1 + w[0]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x1, x2, z, antialiased=True, alpha=0.2)
    ax.scatter( matrixTest[:,0],  matrixTest[:,1],  matrixTest[:,2], marker='^', c='g', label='test set', alpha=1)
    ax.scatter(  matrixTrain[:, 0],   matrixTrain[:, 1],   matrixTrain[:, 2], marker='o', c='r', label='training set', alpha=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('h(x1, x2)')
    plt.show()

def firstdegreeformula(x, w):
    return w[1]*x + w[0]

def seconddegreeformula(x1, x2, w):
    return w[2]*x2 + w[1] * x1 + w[0]

def meansquarederror(matrix, w):
    X, y = getXandY(matrix)
    mse = (np.linalg.norm((X.dot(w[range(1,len(w))])-y))**2)/len(y)
    print(mse)


#array = addOne(readCSV('datasets/regression/reg-1d-test.csv'))
matrixTrain = readCSV('datasets/regression/reg-1d-train.csv')
matrixTrain = addOne(matrixTrain)
X, y = getXandY(matrixTrain)
w = linearRegression(X, y)
matrixTest = readCSV('datasets/regression/reg-1d-test.csv')
matrixTrain = readCSV('datasets/regression/reg-1d-train.csv')
graph2d(firstdegreeformula, np.arange(0, 1, 0.01), matrixTest, matrixTrain, w)
print('1D weights')
print(w)
print('MSE test, train 1D')
meansquarederror(matrixTest, w)
meansquarederror(matrixTrain, w)

matrixTrain = readCSV('datasets/regression/reg-2d-train.csv')
matrixTrain = addOne(matrixTrain)
X, y = getXandY(matrixTrain)
w = linearRegression(X, y)
print('2D weights')
print(w)
matrixTest = readCSV('datasets/regression/reg-2d-test.csv')
matrixTrain = readCSV('datasets/regression/reg-2d-train.csv') #I'm lazy, so instead of removing/ignoring the column of 1's, I just re-read the file
graph3d(seconddegreeformula, np.arange(0, 1, 0.01), np.arange(0,1, 0.01),  matrixTest,  matrixTrain, w)
print('MSE test, train 2D')
meansquarederror(matrixTest, w)
meansquarederror(matrixTrain, w)

