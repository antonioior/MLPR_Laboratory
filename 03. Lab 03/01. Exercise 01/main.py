import numpy as np
import matplotlib.pyplot as plt

#LAB 02
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

#LAB 03
def projection(U, m):
    return U[:,::-1][:,0:m]

def percentageVariance(Eigenvalues):
    eigenvalues = Eigenvalues[::-1]
    print("Eigenvalues sorted:\n", eigenvalues)
    sumEigenvalues = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        sumMEingvalues = np.sum(eigenvalues[:i+1])
        ratio = sumMEingvalues / sumEigenvalues * 100
        print(f"Percentage of variance with m = {i+1}:\n{ratio:.3f}%")
    

def PCA(D, L, C):
    #EIGENVERCTORS AND EIGENVALUES
    s, U = np.linalg.eigh(C)
    print("Eigenvalues s:\n", s)
    print("Eigenvectors U:\n", U)

    #PERCENTAGE
    percentageVariance(s)

    P = projection(U, 2)
    print("P:\n", P)
    
    #Another way to obtain the same thing, across SVD
    # U, s, Vh = np.linalg.svd(C)
    # print("U svd:\n", U)
    # print("s svd:\n", s)
    # print("Vh svd:\n", Vh)
    # P = U[:,0:1]
    # print("P:\n", P)

    resultEigenvector = np.load('IRIS_PCA_matrix_m4.npy')
    print("Eigenvector prof:\n", resultEigenvector)

    #PROJECTION
    DP = np.dot(P.T, D)
    print("DP shape is:\n", DP.shape)
    plot(DP[:, L == 0], DP[:, L == 1], DP[:, L == 2], 0, 1, "", "")
    plt.show()


def LDA():
    pass

if __name__ == '__main__':
    #LAB 02
    createGraph = False
    D, L = load('iris.csv')
    properties = ['Sepal Length', 'Sepal Width', 'Petal Lenght', 'Petal Width']

    if createGraph:
        createGraphic(D, L, properties, "without mean")
    

    #STATISTICS
    #MEAN
    mu = D.mean(axis=1)
    muColumn=mcol(mu, D.shape[0])
    muRow=mrow(mu, D.shape[0])

    DC = D - muColumn

    #after closed all without mean it will print that with mean
    if createGraph:
        createGraphic(DC, L, properties, "with mean")
    
    #COVARIANCE
    #print("Covariance matrix with np.cov(DC)\n", np.cov(DC))
    #C = (DC @ DC.T) / (D.shape[1])
    #print("Covariance matrix with formula (DC @ DC.T) / (D.shape[1])\n", C)
    C = (DC.dot(DC.T)) / (D.shape[1])
    print("Covariance matrix with dot product (DC.dot(DC.T)) / (D.shape[1])\n", C)
    var = D.var(1)  #variance is the square of std
    std = D.std(1)  
    print("Variance is:\n", mcol(var, D.shape[0]))
    print("Std is:\n", mcol(std, D.shape[0]))
    
    #LAB 03
    #Calculate PCA
    PCA(D, L, C)

    #Calculate LDA
    LDA()

    