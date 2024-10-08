from load import load, loadClassifications
from PCA import PCA
from LDA import LDA
import graph
import utils as ut
import printValue
import numpy as np

if __name__ == '__main__':
    #LAB 02
    #VARIABLES TO PRINT
    createGraph = False
    printValueMain = False

    D, L = load('iris.csv')
    properties = ['Sepal Length', 'Sepal Width', 'Petal Lenght', 'Petal Width']

    if createGraph:
        graph.createGraphic(D, L, properties, "without mean")
    

    #STATISTICS
    #MEAN
    mu = D.mean(axis=1)
    muColumn = ut.mcol(mu, D.shape[0])
    muRow = ut.mrow(mu, D.shape[0])

    DC = D - muColumn

    #after closed all without mean it will print that with mean
    if createGraph:
        graph.createGraphic(DC, L, properties, "with mean")
    
    #COVARIANCE
    #print("Covariance matrix with np.cov(DC)\n", np.cov(DC))
    #C = (DC @ DC.T) / (D.shape[1])
    #print("Covariance matrix with formula (DC @ DC.T) / (D.shape[1])\n", C)
    C = (DC.dot(DC.T)) / (D.shape[1])
    var = D.var(1)  #variance is the square of std
    std = D.std(1)  
    
    
    if printValueMain:
        print("MAIN - RESULT")
        print(f"    Covariance matrix with dot product (DC.dot(DC.T)) / (D.shape[1])")
        printValue.printMatrix(C)
        print(f"    Variance is:")
        printValue.printMatrix(ut.mcol(var, D.shape[0]))
        print(f"    Std is:")
        printValue.printMatrix(ut.mcol(std, D.shape[0]))

    #LAB 03
    #PCA
    #Calculate PCA
    dataProjectedPCA, _ = PCA(D, L, printResults = False)

    #LDA
    #Calculate LDA
    dataProjectedLDA, _ = LDA(D, L, printResults = False)

    #Print data
    graph.createGraphicPCA_LDA(L, dataProjectedPCA, dataProjectedLDA)

    #PCA LDA FOR CLASSIFICATION
    #Upload data only of versicolor and virginica
    D, L = loadClassifications()
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    dataProjected, W = LDA(DTR, LTR, printResults = False, comment="Classification")
    DTR_lda = np.dot(W.T, DTR)
    DVAL_lda = np.dot(W.T, DVAL)
    graph.createGraphicTrainingLDA(DTR_lda, LTR, DVAL_lda, LVAL)

    #Note part
    ut.calculateError(DTR_lda, LTR, DVAL_lda, LVAL)

    #Startiting apply PCA
    #Convolution on training data
    DTR_pca, P = PCA(DTR, LTR, printResults = False)
    DVAL_pca = np.dot(P.T, DVAL)
    
    #Calculate LDA on DTR of PCA
    dataProjectedLDA, W = LDA(DTR_pca, LTR, printResults = False, comment="Classification after PCA")
    DTR_lda = np.dot(W.T, DTR_pca)
    DVAL_lda = np.dot(W.T, DVAL_pca)
    ut.calculateError(DTR_lda, LTR, DVAL_lda, LVAL)
    graph.createGraphicTrainingLDA(DTR_lda, LTR, DVAL_lda, LVAL)