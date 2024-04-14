#LAB 3
import numpy as np
import scipy as sp
import projectionFunction
import printValue

def computeSb(D, numClass, nSamplesClass, muClass):
    mu = projectionFunction.mcol(D.mean(axis=1), D.shape[0])
    Sb = 0
    for i in range(numClass):
        Sb += nSamplesClass[i] * np.dot((muClass[i] - mu), (muClass[i] - mu).T)
    
    Sb = Sb/D.shape[1]
    return Sb

def computeSw(D,D_class, numClass, nSamplesClass, muClass):
    Sw = 0
    for i in range(numClass):
        DC = D_class[i] - muClass[i]
        Sw += nSamplesClass[i]*np.dot(DC, DC.T) / DC.shape[1]
    Sw = Sw/D.shape[1]
    return Sw

def computeSb_Sw(D, L):
    #determina il valore delle etichette, in caso di PCA e LDA per training
    #ha valore 1 e 2
    valueClass = set(L)

    #num of classes
    numClass = len(set(L))
    
    #separate data by class
    D_class = [D[:, L == i] for i in valueClass]

    #number of samples per class
    nSamplesClass = [D_class[i].shape[1] for i in range(numClass)]
    
    #mean of each class
    mu_class = [projectionFunction.mcol(D_class[i].mean(axis=1), D_class[i].shape[0]) for i in range(numClass)]
    
    Sb = computeSb(D, numClass, nSamplesClass, mu_class)
    Sw = computeSw(D, D_class, numClass, nSamplesClass, mu_class)
    return Sb, Sw

def calculateEigenvalues(Sb, Sw):
    s, U = sp.linalg.eigh(Sb, Sw)

    W = projectionFunction.projection(U, 2)
    #UW, _, _ = np.linalg.svd(W)
    #U = UW[:,0:2]
    #print("U:\n", U)
    
    return s, U, W

def LDA(D, L, printResults = False, comment =""):
    resultEigenvector = np.load("./../Solutions/IRIS_LDA_matrix_m2.npy")

    Sb, Sw = computeSb_Sw(D, L)
    s, U, W = calculateEigenvalues(Sb, Sw)
    
    #PROJECTION
    DP = np.dot(W.T, D)

    if printResults:
        print("LDA - RESULT" + " " + comment)
        print(f"    Sb:")
        printValue.printMatrix(Sb)
        print(f"    Sw:")
        printValue.printMatrix(Sw)
        print(f"    Eigenvalues s:\n\t{s}")
        print(f"    Eigenvectors U:")
        printValue.printMatrix(U)
        print(f"    W:")
        printValue.printMatrix(W)
        print(f"    Eigenvector prof:")
        printValue.printMatrix(resultEigenvector)

    return DP, W