#LAB 3
import numpy as np
import projectionFunction
import graph

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

    P = projectionFunction.projection(U, 2)
    print("P:\n", P)
    
    #Another way to obtain the same thing, across SVD
    # U, s, Vh = np.linalg.svd(C)
    # print("U svd:\n", U)
    # print("s svd:\n", s)
    # print("Vh svd:\n", Vh)
    # P = U[:,0:1]
    # print("P:\n", P)

    resultEigenvector = np.load("./../Solutions/IRIS_PCA_matrix_m4.npy")
    print("Eigenvector PCA prof:\n", resultEigenvector)

    #PROJECTION
    DP = np.dot(P.T, D)
    print("DP shape is:\n", DP.shape)
    graph.plot(DP[:, L == 0], DP[:, L == 1], DP[:, L == 2], 0, 1, "", "")
    graph.plt.show()