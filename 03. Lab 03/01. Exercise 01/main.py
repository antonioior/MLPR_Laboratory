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
    dataProjectedPCA = PCA(D, L, C, printResults = False)

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

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print("LVAL\n", LVAL)
    print("PVAL\n", PVAL)
    difference = np.abs(LVAL - PVAL)
    numOfErr = sum( x != 0 for x in difference)
    print("Num of error\n", numOfErr)