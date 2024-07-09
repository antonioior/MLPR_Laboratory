import numpy as np

from Binary_MVG import Binary_MVG, Binary_Tied
from NB import NB
from PCA import PCA
from printValue import printMatrix
from utils import split_db_2to1, compute_covariance_I_Class, correlationMatrix


def point1(DTR, LTR, DTE, LTE):
    errorRate_MVG, llr_MVG = Binary_MVG(DTR, LTR, DTE, LTE)
    errorRate_Tied, llr_Tied = Binary_Tied(DTR, LTR, DTE, LTE)
    errorRate_NB, llr_NB = NB(DTR, LTR, DTE, LTE)
    mapLlr = {
        "MVG": llr_MVG,
        "TIED": llr_Tied,
        "NB": llr_NB
    }
    return errorRate_MVG, errorRate_Tied, errorRate_NB, mapLlr


def point2(DTR, LTR):
    covarianceIClass = []
    correlationIClass = []
    for i in set(LTR):
        covarianceIClass.append(compute_covariance_I_Class(DTR[:, LTR == i]))
        correlationIClass.append(correlationMatrix(covarianceIClass[i]))
    return covarianceIClass, correlationIClass


def point4(DTR, LTR, DTE, LTE):
    errorRate_MVG_first_four, _ = Binary_MVG(DTR[0:4, :], LTR, DTE[0:4, :], LTE)
    errorRate_Tied_first_four, _ = Binary_Tied(DTR[0:4, :], LTR, DTE[0:4, :], LTE)
    errorRate_NB_first_four, _ = NB(DTR[0:4, :], LTR, DTE[0:4, :], LTE)
    return errorRate_MVG_first_four, errorRate_Tied_first_four, errorRate_NB_first_four


def point5(DTR, LTR, DTE, LTE):
    errorRate_MVG_first_second, _ = Binary_MVG(DTR[0:2, :], LTR, DTE[0:2, :], LTE)
    errorRate_Tied_first_second, _ = Binary_Tied(DTR[0:2, :], LTR, DTE[0:2, :], LTE)
    errorRate_NB_first_second, _ = NB(DTR[0:2, :], LTR, DTE[0:2, :], LTE)

    errorRate_MVG_third_fourth, _ = Binary_MVG(DTR[2:4, :], LTR, DTE[2:4, :], LTE)
    errorRate_Tied_third_fourth, _ = Binary_Tied(DTR[2:4, :], LTR, DTE[2:4, :], LTE)
    errorRate_NB_third_fourth, _ = NB(DTR[2:4, :], LTR, DTE[2:4, :], LTE)

    return (errorRate_MVG_first_second, errorRate_Tied_first_second, errorRate_NB_first_second,
            errorRate_MVG_third_fourth, errorRate_Tied_third_fourth, errorRate_NB_third_fourth)


def point6(DTR, LTR, DTE, LTE):
    Presult = []
    listError = []
    llrPCA = {
        "MVG": [],
        "TIED": [],
        "NB": []
    }
    for i in range(1, 7):
        DTR_pca, P, _ = PCA(DTR, i, printResults=False)
        DTEL_pca = np.dot(P.T, DTE)
        errorMVG, errorTied, errorNB, llr = point1(DTR_pca, LTR, DTEL_pca, LTE)
        Presult.append(P)
        dict = {
            "MVG": errorMVG,
            "Tied": errorTied,
            "NB": errorNB
        }
        llrPCA["MVG"].append({
            "m": i,
            "llr": llr["MVG"]
        })
        llrPCA["TIED"].append({
            "m": i,
            "llr": llr["TIED"]
        })
        llrPCA["NB"].append({
            "m": i,
            "llr": llr["NB"]
        })
        listError.append(dict)
    return Presult, listError, llrPCA


def generativeModels(D, L, printResults=False):
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    errorRate_MVG, errorRate_Tied, errorRate_NB, mapLlr = point1(DTR, LTR, DTE, LTE)
    covarianceIClass, correlationIClass = point2(DTR, LTR)
    errorRate_MVG_first_four, errorRate_Tied_first_four, errorRate_NB_first_four = point4(DTR, LTR, DTE, LTE)
    (errorRate_MVG_first_second, errorRate_Tied_first_second, errorRate_NB_first_second,
     errorRate_MVG_third_fourth, errorRate_Tied_third_fourth, errorRate_NB_third_fourth) = point5(DTR, LTR, DTE, LTE)
    P, errors, mapLlrPCA = point6(DTR, LTR, DTE, LTE)

    if printResults:
        print("MAIN - RESULT OF GENERATIVE MODELS")
        print(f"\tPoint 1")
        print(f"\tError rate for feature 1 to 6")
        print(f"\t\tError rate of Binary MVG: {errorRate_MVG * 100:.2f}%")
        print(f"\t\tError rate of Binary Tied: {errorRate_Tied * 100:.2f}%")
        print(f"\t\tError rate of Naive Bayes: {errorRate_NB * 100:.2f}%")
        print(f"\tPoint 2")
        for i in range(len(covarianceIClass)):
            print(f"\t\tCovariance of class {i}:")
            printMatrix(covarianceIClass[i])
        for i in range(len(correlationIClass)):
            print(f"\t\tCorrelation of class {i}:")
            printMatrix(correlationIClass[i], decimals=2)
        print(f"\tPoint 3 - No result to calculate")
        print(f"\tPoint 4")
        print(f"\tFeature 1 to 4")
        print(f"\t\tError rate of Binary MVG: {errorRate_MVG_first_four * 100:.2f}%")
        print(f"\t\tError rate of Binary Tied: {errorRate_Tied_first_four * 100:.2f}%")
        print(f"\t\tError rate of Naive Bayes: {errorRate_NB_first_four * 100:.2f}%")
        print(f"\tPoint 5")
        print(f"\t\tFeature 1 and 2")
        print(f"\t\tError rate of Binary MVG: {errorRate_MVG_first_second * 100:.2f}%")
        print(f"\t\tError rate of Binary Tied: {errorRate_Tied_first_second * 100:.2f}%")
        print(f"\t\tError rate of Naive Bayes: {errorRate_NB_first_second * 100:.2f}%")
        print(f"\t\tFeature 3 and 4")
        print(f"\t\tError rate of Binary MVG: {errorRate_MVG_third_fourth * 100:.2f}%")
        print(f"\t\tError rate of Binary Tied: {errorRate_Tied_third_fourth * 100:.2f}%")
        print(f"\t\tError rate of Naive Bayes: {errorRate_NB_third_fourth * 100:.2f}%")
        print(f"\tPoint 6 - PCA + MVG")
        for i in range(len(P)):
            print(f"\t\tPCA + MVG m = {i + 1}")
            printMatrix(P[i])
            print(f"\t\tError rate of Binary MVG: {errors[i]["MVG"] * 100:.2f}%")
            print(f"\t\tError rate of Binary Tied: {errors[i]["Tied"] * 100:.2f}%")
            print(f"\t\tError rate of Naive Bayes: {errors[i]["NB"] * 100:.2f}%")

    return LTE, mapLlr, mapLlrPCA
