#LAB 3
import numpy as np
import utils as ut
import printValue

def percentageVariance(Eigenvalues):
    eigenvalues = Eigenvalues[::-1]
    #print("Eigenvalues sorted:\n", eigenvalues)
    sumEigenvalues = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        sumMEingvalues = np.sum(eigenvalues[:i+1])
        ratio = sumMEingvalues / sumEigenvalues * 100
        #print(f"Percentage of variance with m = {i+1}:\n{ratio:.3f}%")
    

def PCA(D, L, printResults = False):
    resultEigenvector = np.load("./../Solutions/IRIS_PCA_matrix_m4.npy")
    
    mu = D.mean(axis=1)
    DC = D - ut.mcol(mu, D.shape[0])
    C = (DC.dot(DC.T)) / (D.shape[1])

    #EIGENVERCTORS AND EIGENVALUES
    s, U = np.linalg.eigh(C)
    

    #PERCENTAGE
    resultPercentageVariange = percentageVariance(s)

    P = ut.projection(U, 2)
    
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
        print(f"    Eigenvalues s:\n\t{s}")
        print(f"    Eigenvectors U:")
        printValue.printMatrix(U)
        print(f"    Projection P:")
        printValue.printMatrix(P)
        print(f"    Eigenvector PCA prof:")
        printValue.printMatrix(resultEigenvector)

    return DP, P