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

def scatterData(matrix):
    class0 = np.empty((0,2), float)
    class1 = np.empty((0,2), float)
    for row in matrix:
        if(row[2] < 1):
            class0 = np.append(class0, np.array([[row[0], row[1]]]), axis=0)
        else:
            class1 = np.append(class1, np.array([[row[0], row[1]]]), axis=0)
    plt.scatter(class0[:,0], class0[:,1], marker='^', c='g', alpha=1)
    fig = plt.scatter(class1[:,0], class1[:,1], marker='o', c='b', alpha=1)
    return fig

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

#def sigma(z):
#    return 1/(1+ math.exp(-z))

def decisionBoundary(w, x):
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]

    y = -((w0 + x*w1)/w2)
    #print y
    return y

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

#Train
X, y = getXandY(matrix)
w = [0.1, -0.5, 0.3]
steps = 1000
learningrate = 0.00000001
w = train(X, y, w, steps, learningrate)
print(w)
matrix = readCSV('datasets/classification/cl-test-2.csv')
X, y = getXandY(matrix)
scatter = scatterData(matrix)
x = np.array(np.arange(0,1,0.01))
boundary = decisionBoundary(w, x)
#plt.plot(x, boundary)
plt.show()



#Test
matrix = readCSV('datasets/classification/cl-test-1.csv')
matrix = addOne(matrix)
X, y = getXandY(matrix)

prediction, res = classify(X, w)
resultmatrix = np.column_stack((y, prediction, res))
correct = 0
for i in range(0, len(prediction)):
    if(prediction[i] == y[i]):
        correct += 1

print(resultmatrix)
print(correct)