# ALL LAB 3 SOLUTIONS
import numpy as np

import graph
import utils as ut
from LDA import LDA
from PCA import PCA


def PCA_LDA(D, L, createGraph=False, printResults=False):
    # Calculate PCA
    dataProjectedPCA, _, ratio = PCA(D, 6, printResults=printResults)
    if createGraph:
        graph.representRatio(ratio)

    # Calculate LDA
    dataProjectedLDA, _ = LDA(D, L, printResults=printResults)

    # Print data
    if createGraph:
        graph.createGraphicPCA_LDA(L, dataProjectedPCA, dataProjectedLDA)

    # PCA and LDA for classification
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    dataProjected, W = LDA(DTR, LTR, printResults=printResults, comment="Classification")
    DTR_lda = np.dot(W.T, DTR)
    DVAL_lda = np.dot(W.T, DVAL)
    if createGraph:
        graph.createGraphicTrainingLDA(DTR_lda, LTR, DVAL_lda, LVAL, comment="Classification")

    # Note part
    ut.calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults=printResults)

    # Startiting apply PCA
    # Convolution on training data
    DTR_pca, P, _ = PCA(DTR, 6, printResults=printResults)
    DVAL_pca = np.dot(P.T, DVAL)

    # Calculate LDA on DTR of PCA
    dataProjectedLDA, W = LDA(DTR_pca, LTR, printResults=printResults, comment="Classification after PCA")
    DTR_lda = np.dot(W.T, DTR_pca)
    DVAL_lda = np.dot(W.T, DVAL_pca)
    ut.calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults=printResults)
    if createGraph:
        graph.createGraphicTrainingLDA(DTR_lda, LTR, DVAL_lda, LVAL, comment="With PCA")
