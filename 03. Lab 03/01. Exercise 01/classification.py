from load import loadClassifications
from LDA import LDA
import numpy as np
import matplotlib.pyplot as plt

#LAB 03
def split_db_2to1(D, L, seed = 0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def classification(printResults = False):
    D, L = loadClassifications()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    dataProjected, W = LDA(DTR, LTR, printResults = printResults, comment="Classification")
    projectedValueT = np.dot(W.T, DTR)
    projectedValueV = np.dot(W.T, DVAL)
    
    return projectedValueT, LTR, projectedValueV, LVAL