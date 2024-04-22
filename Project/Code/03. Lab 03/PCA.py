# LAB 03
import numpy as np
import utils as ut
import printValue


def percentageVariance(Eigenvalues):
    eigenvalues = Eigenvalues[::-1]
    ratio = []
    # print("Eigenvalues sorted:\n", eigenvalues)
    sumEigenvalues = np.sum(eigenvalues)
    for i in range(0, len(eigenvalues)):
        sumMEingvalues = np.sum(eigenvalues[:i+1])
        ratio.append(sumMEingvalues / sumEigenvalues * 100)
        # print(f"Percentage of variance with m = {i+1}:\n{ratio[i]:.3f}%")
    return ratio    


def PCA(D, L, m, printResults=False):
    mu = D.mean(axis=1)
    DC = D - ut.mcol(mu, D.shape[0])
    C = (DC.dot(DC.T)) / (D.shape[1])
    
    # EIGENVERCTORS AND EIGENVALUES
    s, U = np.linalg.eigh(C)

    # PERCENTAGE
    ratio = percentageVariance(s)
    P = ut.projection(U, m)

    # PROJECTION
    DP = np.dot(P.T, D)
    
    if printResults:
        print("PCA - RESULTS")
        print(f"    Eigenvalues s in ascending order:\n\t{s}")
        print("    Eigenvectors U for eigenvalue in ascendig ordere:")
        printValue.printMatrix(U)
        print("    Projection P is the matrix of eigenvectors in descending order of eigenvalues."
              " This is the order to take:")
        printValue.printMatrix(P)

    return DP, P, ratio
        