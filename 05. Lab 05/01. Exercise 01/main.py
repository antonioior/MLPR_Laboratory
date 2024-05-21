import numpy as np

from Binary_MVG import Binary_MVG, Binary_Tied
from LDA import LDA
from MVG import MVG
from NB import NB
from TCG import TCG
from load import load, loadOnlyVersicolarAndVirginica
from utils import split_db_2to1, calculateError

if __name__ == "__main__":
    printData = False
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # First model Multivariate Gaussian Classifier
    MVG(DTR, LTR, DTE, LTE, printData=False)
    NB(DTR, LTR, DTE, LTE, printData=False)
    TCG(DTR, LTR, DTE, LTE, printData=False)

    # Only on Versicolar and Virginica
    D, L = loadOnlyVersicolarAndVirginica()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    Binary_MVG(DTR, LTR, DTE, LTE, printData=False)
    Binary_Tied(DTR, LTR, DTE, LTE, printData=True)

    dataProjected, W = LDA(DTR, LTR, printResults=False, comment="Classification")
    DTR_lda = np.dot(W.T, DTR)
    DTE_lda = np.dot(W.T, DTE)
    calculateError(DTR_lda, LTR, DTE_lda, LTE, printResults=True)
