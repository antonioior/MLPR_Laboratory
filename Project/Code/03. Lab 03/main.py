from load import load
from PCA import PCA
from LDA import LDA
import numpy as np
import utils as ut
import graph
import printValue


if __name__ == "__main__":

    # VARIABLES TO PRINT
    createGraph = False
    
    # LAB 02
    D, L = load('trainData.txt')
    properties = ["Features 1", "Features 2", "Features 3", "Features 4", "Features 5", "Features 6"]
    
    mu = D.mean(axis=1)
    muColumn = ut.mcol(mu, D.shape[0])
    DC = D - muColumn

    var = D.var(1)
    std = D.std(1)  # square of variance
    varColumn = ut.mcol(var, D.shape[0])
    stdColumn = ut.mcol(std, D.shape[0])
    
    if createGraph:
        # POINT 1, graphics only of the first two features
        graph.createGraphic(D, L, properties, 0, 2, "without mean")
        graph.createGraphic(DC, L, properties, 0, 2, "with mean")
        graph.plt.show()

        # POINT 2, graphics of second and third the features.
        # The figure will be created only when you close the previous one
        graph.createGraphic(D, L, properties, 2, 4, "without mean")
        graph.createGraphic(DC, L, properties, 2, 4, "with mean")
        graph.plt.show()
    
        # POINT 3, graphics of the last two features. The figure will be created only when you close the previous one
        graph.createGraphic(D, L, properties, 4, 6, "without mean")
        graph.createGraphic(DC, L, properties, 4, 6, "with mean")
        graph.plt.show()
    
    # LAB 03
    # Covariance
    C = (DC.dot(DC.T)) / (D.shape[1])
    var = D.var(1)  # variance is the square of std
    std = D.std(1)  
    
    printValue.printDataMain(muColumn, varColumn, C, var, std, D, printData=False)

    # Calculate PCA
    dataProjectedPCA, _, ratio = PCA(D, L, 6, printResults=False)
    graph.representRatio(ratio)

    # Calculate LDA
    dataProjectedLDA, _ = LDA(D, L, printResults=False)

    # Print data
    graph.createGraphicPCA_LDA(L, dataProjectedPCA, dataProjectedLDA)

    # PCA and LDA for classification
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    dataProjected, W = LDA(DTR, LTR, printResults=False, comment="Classification")
    DTR_lda = np.dot(W.T, DTR)
    DVAL_lda = np.dot(W.T, DVAL)
    graph.createGraphicTrainingLDA(DTR_lda, LTR, DVAL_lda, LVAL, comment="")

    # Note part
    ut.calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults=True)

    # Startiting apply PCA
    # Convolution on training data
    DTR_pca, P, _ = PCA(DTR, LTR, printResults=False, m=1)
    DVAL_pca = np.dot(P.T, DVAL)
    
    # Calculate LDA on DTR of PCA
    dataProjectedLDA, W = LDA(DTR_pca, LTR, printResults=False, comment="Classification after PCA")
    DTR_lda = np.dot(W.T, DTR_pca)
    DVAL_lda = np.dot(W.T, DVAL_pca)
    ut.calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults=False)
    graph.createGraphicTrainingLDA(DTR_lda, LTR, DVAL_lda, LVAL, comment="With PCA")
