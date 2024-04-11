#LAB 03
import numpy as np
import graph
import projectionFunction
import printValue

def percentageVariance(Eigenvalues):
    eigenvalues = Eigenvalues[::-1]
    #print("Eigenvalues sorted:\n", eigenvalues)
    sumEigenvalues = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        sumMEingvalues = np.sum(eigenvalues[:i+1])
        ratio = sumMEingvalues / sumEigenvalues * 100
        #print(f"Percentage of variance with m = {i+1}:\n{ratio:.3f}%")

def PCA(D, L, C, printResults = False):
    s, U = np.linalg.eigh(C)

    #PERCENTAGE
    percentageVariance(s)
    P = projectionFunction.projection(U, len(s))

    #PROJECTION
    DP = np.dot(P.T, D)
    
    if printResults:
        print("PCA - RESULTS")
        print(f"    Eigenvalues s in ascending order:\n\t{s}")
        print("    Eigenvectors U for eigenvalue in ascendig ordere:")
        printValue.printMatrix(U)
        print("    Projection P is the matrix of eigenvectors in descending order of eigenvalues. This is the order to take:")
        printValue.printMatrix(P)

    return DP
        