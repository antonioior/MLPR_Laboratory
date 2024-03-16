import numpy as np
import matplotlib.pyplot as plt

def load(fileName):
    data = open(fileName, 'r',)
    D = None
    L = np.array([], dtype=int)
    for row in data:
        values = row.strip().split(',')

        columnProperty = np.array(values[0:4], dtype=float).reshape(4,1)
        flowerType = -1
        if values[4] == 'Iris-setosa':
            flowerType = 0
        elif values[4] == 'Iris-versicolor':
            flowerType = 1
        elif values[4] == 'Iris-virginica':
            flowerType = 2

        if D is None:
            D = columnProperty
        else:
            D = np.append(D, columnProperty, axis=1)

        L = np.append(L, flowerType)

    return D, L 

def createGraphic(data, labels, properties, comment):
    data0 = data[:, labels == 0]
    data1 = data[:, labels == 1]
    data2 = data[:, labels == 2]
    for i in range(0, len(properties)):
        for j in range(0, len(properties)):
            if i == j:
                plt.figure(properties[i] + " " + comment)
                hist(data0, data1, data2, i, properties[i])
            else:
                plt.figure(properties[i] + " " + properties[j] + " " + comment)
                plot(data0, data1, data2, i, j, properties[i], properties[j])
    plt.show()


def hist(D0, D1, D2, property, features):
    plt.hist(D0[property,:], density=True, label='Setosa')
    plt.hist(D1[property,:], density=True, label='Versicolor')
    plt.hist(D2[property,:], density=True, label='Virginica')
    plt.xlabel(features)
    plt.legend()

def plot(D0, D1, D2, propertyX, propertyY, featureX, featureY):
    plt.plot(D0[propertyX,:], D0[propertyY,:], 'o', label='Setosa')
    plt.plot(D1[propertyX,:], D1[propertyY,:], 'o', label='Versicolor')
    plt.plot(D2[propertyX,:], D2[propertyY,:], 'o', label='Virginica')
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.legend()

def mcol(mu, shape):
    return mu.reshape(shape, 1)

def mrow(mu, shape):
    return mu.reshape(1, shape)

if __name__ == '__main__':
    D, L = load('iris.csv')
    properties = ['Sepal Length', 'Sepal Width', 'Petal Lenght', 'Petal Width']

    createGraphic(D, L, properties, "without mean")
    

    #STATISTICS
    #MEAN
    mu = D.mean(axis=1)
    muColumn=mcol(mu, D.shape[0])
    muRow=mrow(mu, D.shape[0])

    DC = D - muColumn

    #after closed all without mean it will print that with mean
    createGraphic(DC, L, properties, "with mean")
    
    #COVARIANCE
    C = (DC @ DC.T) / float(D.shape[1])
    print(np.cov(DC))
    var = D.var(1)
    std = D.std(1)  #square of variance
    print(mcol(var, D.shape[0]))
    print(mcol(std, D.shape[0]))