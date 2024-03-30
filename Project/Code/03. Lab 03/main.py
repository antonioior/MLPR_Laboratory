import numpy as np
import matplotlib.pyplot as plt

#LAB 02
#Function to load the data from the file
def load(fileName):
    data = open(fileName, 'r',)
    D = None
    L = np.array([], dtype=int)
    for row in data:
        values = row.strip().split(',')

        columnProperty = mcol(np.array(values[0:6], dtype=float), 6)

        if D is None:
            D = columnProperty
        else:
            D = np.append(D, columnProperty, axis=1)

        L = np.append(L, int(values[6]))

    return D, L 

#Function to create a column vector
def mcol(data, shape):
    return data.reshape(shape, 1)

#Function to create a row vector
def mrow(data, shape):
    return data.reshape(1, shape)

#Functoin to create the graphics
def createGraphic(data, labels, properties, initialProperty, finalProperty, comment):
    data0 = data[:, labels == 0]
    data1 = data[:, labels == 1]
    plt.figure("Graphics of the properties " + properties[initialProperty] + " and " + properties[finalProperty-1] + " " + comment, figsize=(10, 10))
    count = 0
    for i in range(initialProperty, finalProperty):
        for j in range(initialProperty, finalProperty):
            count+=1
            if i == j:
                plt.subplot(finalProperty-initialProperty, finalProperty-initialProperty, count)
                hist(data0, data1, i, properties[i])
            else:
               plt.subplot(finalProperty-initialProperty, finalProperty-initialProperty, count)
               scatter(data0, data1, i, j, properties[i], properties[j])
    plt.tight_layout()

#Function called by createGraphic to create the histogram
def hist(D0, D1, property, features):
    plt.hist(D0[property,:], density=True, label='False', alpha=0.5)
    plt.hist(D1[property,:], density=True, label='True', alpha=0.5)
    plt.xlabel(features)
    plt.legend()

#Function called by createGraphic to create the scatter plot
def scatter(D0, D1, propertyX, propertyY, featureX, featureY):
    plt.scatter(D0[propertyX,:], D0[propertyY,:], label='False', alpha=0.5)
    plt.scatter(D1[propertyX,:], D1[propertyY,:], label='True', alpha=0.5)
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.legend()

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
    print("Covarience shape C:\n", C.shape)
    s, U = np.linalg.eigh(C)
    print("Eigenvalues:\n", s)
    print("Eigenvectors:\n", U)

    #PERCENTAGE
    percentageVariance(s)

    P = projection(U, 6)
    print("Projection matrix P:\n", P)

    #PROJECTION
    DP = np.dot(P.T, D)
    print("DP shape is:\n", DP.shape)
    scatter(DP[:, L == 0], DP[:, L == 1], 1, 4, "", "")
    plt.show()

if __name__ == "__main__":
    #LAB 02
    createGraph = False
    D, L = load('trainData.txt')
    properties =["Features 1", "Features 2", "Features 3", "Features 4", "Features 5" ,"Features 6"]
    
    mu = D.mean(axis=1)
    muColumn = mcol(mu, D.shape[0])
    DC = D - muColumn

    var = D.var(1)
    std = D.std(1)  #square of variance
    varColumn = mcol(var, D.shape[0])
    stdColumn = mcol(std, D.shape[0])

    print("Mean of the properties:\n", muColumn)
    print("Variance of the properties:\n", varColumn)

    
    #POINT 1, graphics only of the first two features
    if createGraph:
        createGraphic(D, L, properties, 0,2, "without mean")
        createGraphic(DC, L, properties, 0, 2, "with mean")
        plt.show()
    print("Mean of the first two properties:\n", muColumn[0:2, :])
    print("Variance of the first two properties:\n", varColumn[0:2, :])

    #POINT 2, graphics of second and third the features. The figure will be created only when you close the previous one
    if createGraph:
        createGraphic(D, L, properties, 2, 4, "without mean")
        createGraphic(DC, L, properties, 2, 4, "with mean")
        plt.show()
    print("Mean of the properties 3 and 4:\n", muColumn[2:4, :])
    print("Variance of the properties 3 and 4:\n", varColumn[2:4, :])

    #POINT 3, graphics of the last two features. The figure will be created only when you close the previous one
    if createGraph:
        createGraphic(D, L, properties, 4, 6, "without mean")
        createGraphic(DC, L, properties, 4, 6, "with mean")
        plt.show()
    print("Mean of the properties 5 and 6:\n", muColumn[4:6, :])
    print("Variance of the properties 5 and 6:\n", varColumn[4:6, :])

    #LAB 03
    #Covariance
    C = (DC.dot(DC.T)) / (D.shape[1])
    print("Covariance matrix with dot product (DC.dot(DC.T)) / (D.shape[1])\n", C)
    var = D.var(1)  #variance is the square of std
    std = D.std(1)  
    print("Variance is:\n", mcol(var, D.shape[0]))
    print("Std is:\n", mcol(std, D.shape[0]))
    #Calculate PCA
    PCA(D, L, C)