import graph
import printValue
import utils as ut
from Calibration import calibration
from Configuration import MainConfiguration
from GMM import GMM
from LinearSVM import linearSVM
from LogisticRegression import logisticRegression
from PCA_LDA import PCA_LDA
from PolinomialSVM import polinomialSVM
from RadialSVM import radialSVM
from densityEstimation import densityEstimation
from generativeModels import generativeModels
from load import load
from utils import split_db_2to1
from Evaluation import evaluation

if __name__ == "__main__":
    trainLogisticRegression = False
    trainSVM = False
    trainGMM = False
    calibrationRun = True
    # VARIABLES TO PRINT
    createGraph = False

    # LAB 02 - COMPLETED
    D, L = load('Data/trainData.txt')
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

    # Covariance
    C = (DC.dot(DC.T)) / (D.shape[1])
    var = D.var(1)  # variance is the square of std
    std = D.std(1)
    printValue.printDataMain(muColumn, varColumn, C, var, std, D, printData=False)

    # LAB 03 - COMPLETED
    # PCA_LDA
    PCA_LDA(D, L, createGraph=False, printResults=False)

    # LAB 04 - COMPLETED
    # DENSITY ESTIMATION
    densityEstimation(D, L, createGraph=False, printResults=False, comment="")

    # LAB 05 - COMPLETED
    # GENERATIVE MODELS
    LTE, mapLlr, mapLlrPCA = generativeModels(D, L, printResults=False)

    # LAB 06 - NO PROJECT PART

    # LAB 07 - COMPLETED
    MainConfiguration(LTE, mapLlr, mapLlrPCA, printResults=False)

    # LAB 08  - COMPLETED
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    if trainLogisticRegression:
        logisticRegression(DTR, LTR, DVAL, LVAL)

    # LAB 09 - COMPLETED
    if trainSVM:
        linearSVM(DTR, LTR, DVAL, LVAL, printResult=True, titleGraph="Linear SVM")
        polinomialSVM(DTR, LTR, DVAL, LVAL, printResult=True, titleGraph="Polynomial SVM")
        radialSVM(DTR, LTR, DVAL, LVAL, printResult=True, titleGraph="Radial SVM")

    # LAB 10 - COMPLETED
    if trainGMM:
        GMM(DTR, LTR, DVAL, LVAL, printResults=True)

    # LAB 11
    if calibrationRun:
        qlr, svm, gmm = calibration(DTR, LTR, DVAL, LVAL, printResult=True)
        evalData, evalLabels = load('Data/evalData.txt')
        pT = 0.1
        prior_Cals = [0.1, 0.5, 0.9]
        for prior_Cal in prior_Cals:
            evaluation(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, prior_Cal, printResult=True)
