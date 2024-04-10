#LAB 3
import numpy as np
import scipy as sp
import projectionFunction
import graph

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
    #num of classes
    numClass = L.max() + 1
    
    #separate data by class
    D_class = [D[:, L == i] for i in range(numClass)]
    
    #number of samples per class
    nSamplesClass = [D_class[i].shape[1] for i in range(numClass)]
    
    #mean of each class
    mu_class = [projectionFunction.mcol(D_class[i].mean(axis=1), D_class[i].shape[0]) for i in range(numClass)]
    
    Sb = computeSb(D, numClass, nSamplesClass, mu_class)
    Sw = computeSw(D, D_class, numClass, nSamplesClass, mu_class)
    print("Sb:\n", Sb)
    print("Sw:\n", Sw)
    return Sb, Sw

def calculateEigenvalues(Sb, Sw):
    s, U = sp.linalg.eigh(Sb, Sw)
    print("Eigenvalues s:\n", s)
    print("Eigenvectors U:\n", U)
    W = projectionFunction.projection(U, 2)
    print("W:\n", W)
    #UW, _, _ = np.linalg.svd(W)
    #U = UW[:,0:2]
    #print("U:\n", U)
    resultEigenvector = np.load("./../Solutions/IRIS_LDA_matrix_m2.npy")
    print("Eigenvector prof:\n", resultEigenvector)
    return W

def LDA(D, L):
    Sb, Sw = computeSb_Sw(D, L)
    W = calculateEigenvalues(Sb, Sw)
    
    #PROJECTION
    DP = np.dot(W.T, D)
    print("DP shape is:\n", DP.shape)
    graph.plot(DP[:, L == 0], DP[:, L == 1], DP[:, L == 2], 0, 1, "", "")
    graph.plt.show()