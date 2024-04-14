#LAB 02
import numpy as np
import sklearn
import sklearn.datasets

#Load all data
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

#LAB 03
def loadClassifications():
    DIris, LIris = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    return D, L