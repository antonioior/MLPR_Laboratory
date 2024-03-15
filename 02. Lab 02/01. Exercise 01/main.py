import numpy as np
import matplotlib.pyplot as plt

def load(fileName):
    data = open(fileName, 'r',)
    D = None
    L = np.array([], dtype=int)
    for row in data:
        values = row.split(',')

        columnProperty = np.array(values[0:4], dtype=float).reshape(4,1)
        flowerType = -1
        if values[4] == 'Iris-setosa\n':
            flowerType = 0
        elif values[4] == 'Iris-versicolor\n':
            flowerType = 1
        elif values[4] == 'Iris-virginica\n':
            flowerType = 2

        if D is None:
            D = columnProperty
        else:
            D = np.append(D, columnProperty, axis=1)

        L = np.append(L, flowerType)

    return D, L 

def plot(D, L, row, features):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]
    plt.hist(D0[row,:], density=True, label='Setosa')
    plt.hist(D1[row,:], density=True, label='Versicolor')
    plt.hist(D2[row,:], density=True, label='Virginica')
    plt.xlabel(features)
    plt.legend()


if __name__ == '__main__':
    D, L = load('iris.csv')
    properties = ['Sepal Length', 'Sepal Width', 'Petal Lenght', 'Petal Width']
    
    for property in range(0, len(properties)):
        plt.figure(properties[property])
        plot(D, L, property, properties[property])

    plt.show()