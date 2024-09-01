import numpy as np

from BinaryLogisticRegression import BinaryLogisticRegression
from PCA import PCA
from PriorWeightedBinLogReg import PriorWeightedBinLogReg
from QuadraticLogisticRegression import QuadraticLogisticRegression
from utils import z_normalizeOther, z_normalizeTraining


def logisticRegression(DTR, LTR, DVAL, LVAL):
    DVALZNormalized = z_normalizeOther(DTR, DVAL)
    DTRZNormalized = z_normalizeTraining(DTR)

    # Binary Logistic Regression
    BinaryLogisticRegression(
        DTR=DTR,
        LTR=LTR,
        DVAL=DVAL,
        LVAL=LVAL,
        titleGraph="Binary Logistic Regression",
        printResult=False)
    BinaryLogisticRegression(
        DTR=DTRZNormalized,
        LTR=LTR,
        DVAL=DVALZNormalized,
        LVAL=LVAL,
        titleGraph="Binary Logistic Regression - Z Normalized",
        printResult=False
    )

    # Binary Logistic Regression with 50 sample
    BinaryLogisticRegression(
        DTR=DTR[:, ::50],
        LTR=LTR[::50],
        DVAL=DVAL,
        LVAL=LVAL,
        titleGraph="Binary Logistic Regression with 50 sample",
        printResult=False)
    BinaryLogisticRegression(
        DTR=DTRZNormalized[:, ::50],
        LTR=LTR[::50],
        DVAL=DVALZNormalized,
        LVAL=LVAL,
        titleGraph="Binary Logistic Regression with 50 sample - Z Normalized",
        printResult=False)

    # Prior Weighted Binary Logistic Regression
    PriorWeightedBinLogReg(
        DTR=DTR,
        LTR=LTR,
        DVAL=DVAL,
        LVAL=LVAL,
        titleGraph="Binary Logistic Regression with prior weighted",
        printResult=False)

    # Quadratic Logistic Regression
    QuadraticLogisticRegression(
        DTR=DTR,
        LTR=LTR,
        DVAL=DVAL,
        LVAL=LVAL,
        titleGraph="Quadratic Logistic Regression",
        printResult=False)

    # WithPCA and z
    normalize = True
    for i in range(1, 7):
        if normalize:
            DVAL = z_normalizeOther(DTR, DVAL)
            DTR = z_normalizeTraining(DTR)
        DTR_pca, P, _ = PCA(DTR, i, printResults=False)
        DVAL_pca = np.dot(P.T, DVAL)
        print(f"PCA with m = {i}")
        BinaryLogisticRegression(
            DTR=DTR_pca,
            LTR=LTR,
            DVAL=DVAL_pca,
            LVAL=LVAL,
            titleGraph=f"Binary Logistic Regression with PCA m = {i} {"with z-normalize" if normalize else ""}",
            printResult=False)
        BinaryLogisticRegression(
            DTR=DTR_pca[:, ::50],
            LTR=LTR[::50],
            DVAL=DVAL_pca,
            LVAL=LVAL,
            titleGraph=f"Binary Logistic Regression with PCA m = {i} with 50 sample {"with z-normalize" if normalize else ""}",
            printResult=False)
        PriorWeightedBinLogReg(
            DTR=DTR_pca,
            LTR=LTR,
            DVAL=DVAL_pca,
            LVAL=LVAL,
            titleGraph=f"Binary Logistic Regression with PCA m = {i} with prior weighted{"with z-normalize" if normalize else ""}",
            printResult=False)
        QuadraticLogisticRegression(
            DTR=DTR_pca,
            LTR=LTR,
            DVAL=DVAL_pca,
            LVAL=LVAL,
            titleGraph=f"Quadratic Logistic Regression with PCA m = {i} {"with z-normalize" if normalize else ""}",
            printResult=False)
