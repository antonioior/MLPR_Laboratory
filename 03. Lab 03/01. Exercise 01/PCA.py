#LAB 3
import numpy as np
import projectionFunction
import graph

def percentageVariance(Eigenvalues):
    eigenvalues = Eigenvalues[::-1]
    #print("Eigenvalues sorted:\n", eigenvalues)
    sumEigenvalues = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        sumMEingvalues = np.sum(eigenvalues[:i+1])
        ratio = sumMEingvalues / sumEigenvalues * 100
        #print(f"Percentage of variance with m = {i+1}:\n{ratio:.3f}%")
    

def PCA(D, L, C, printResults = False):
    resultEigenvector = np.load("./../Solutions/IRIS_PCA_matrix_m4.npy")

    #EIGENVERCTORS AND EIGENVALUES
    s, U = np.linalg.eigh(C)
    

    #PERCENTAGE
    resultPercentageVariange = percentageVariance(s)

    P = projectionFunction.projection(U, 2)
    
    #Another way to obtain the same thing, across SVD
    # U, s, Vh = np.linalg.svd(C)
    # print("U svd:\n", U)
    # print("s svd:\n", s)
    # print("Vh svd:\n", Vh)
    # P = U[:,0:1]
    # print("P:\n", P)

    #PROJECTION
    DP = np.dot(P.T, D)

    #PRINT RESULTS
    if printResults:
        print("PCA - RESULTS")
        print(f"\tEigenvalues s:\n\t\t{s}")
        print(f"\tEigenvectors U:\n\t\t{U}")
        print(f"\tProjection P:\n\t\t{P}")
        print(f"Eigenvector PCA prof:\n\t\t{resultEigenvector}")

    return DP