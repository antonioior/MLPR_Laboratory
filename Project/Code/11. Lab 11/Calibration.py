import numpy as np
import scipy.optimize as sp

from QuadraticLogisticRegression import quadraticLogClass
from DCF import bayesError, minDCF, actDCF
from graph import createBayesErrorPlots
from utils import vrow, logpdf_GMM
from SVMClass import SVM
from LBG import trainGMMReturnMinAndActDCF, trainGMMCalibrationReturnMinAndActDCF
from PriorWeightedBinLogReg import priorWeightedLogClass


def calibration(DTR, LTR, DVAL, LVAL, printResult=False):
    K = 5
    priorT = 0.1
    priorCals = [0.1, 0.5, 0.9]
    for priorCal in priorCals:
        calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult)
        calibrationSVM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult)
        # calibrationGMM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult)


# CALIBRATION LOGISTIC REGRESSION
def calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult):
    l = 3.162278e-2
    logRegQDT = quadraticLogClass(DTR, LTR, l)
    score, minDCFWithoutCal, actDCFWithoutCal = logRegQDT.trainReturnMinAndActDCF(DVAL, LVAL, priorT)
    minDCFKFold, actDCFKFold, llrK, labelK = logRegQDT.trainCalibrationReturnMinAndActDCF(K, priorCal, priorT, score,
                                                                                          LVAL)
    if printResult:
        print(f"priorCal {priorCal}")
        print("\tCALIBRATION QUADRATIC LOGISTIC REGRESSION")
        print(f"\tLambda: {l}")
        printData(minDCFWithoutCal, actDCFWithoutCal, minDCFKFold, actDCFKFold, score, LVAL, llrK, labelK,
                  f"QLR - calibration validation priorCal = {priorCal}", "b")


# CALIBRATION SVM
def calibrationSVM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult=False):
    C = 100
    gamma = 0.1
    KRadial = 1
    radialSVM = SVM(DTR, LTR, C, KRadial, "radial", gamma=gamma)
    score, minDCFWithoutCal, actDCFWithoutCal = radialSVM.tainRadialReturnMinAndActDCF(DVAL, LVAL,
                                                                                       priorT)
    minDCFKFold, actDCFKFold, llrK, labelK = radialSVM.trainCalibrationReturnMinAndActDCF(K, priorCal, priorT, score,
                                                                                          LVAL)

    if printResult:
        print("\tRESULT FOR CALIBRATION SVM")
        print(f"\tC: {C}, \u03B3: {gamma}, KRadial: {KRadial}")
        printData(minDCFWithoutCal, actDCFWithoutCal, minDCFKFold, actDCFKFold, score, LVAL, llrK, labelK,
                  f"SVM - calibration validation priorCal = {priorCal}", "orange")


# CALIBRATION GMM
def calibrationGMM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult=False):
    covType = "diagonal"
    componentGMM0 = 8
    componentGMM1 = 32
    psi = 0.01
    alpha = 0.1
    sllrWithoutCal, minDCFWithoutCal, actDCFWithoutCal = trainGMMReturnMinAndActDCF(DTR, LTR, DVAL, LVAL,
                                                                                    priorT, alpha,
                                                                                    componentGMM0,
                                                                                    componentGMM1, psi,
                                                                                    covType)
    minDCFKFold, actDCFKFold, llrK, labelK = trainGMMCalibrationReturnMinAndActDCF(K, priorCal, priorT,
                                                                                   sllrWithoutCal, LVAL)

    if printResult:
        print(f"\tRESULT FOR CALIBRATION GMM")
        print(
            f"\tcovType: {covType}, component0 = {componentGMM0}, component1 = {componentGMM1}, psi: {psi}, alpha: {alpha}")
        printData(minDCFWithoutCal, actDCFWithoutCal, minDCFKFold, actDCFKFold, sllrWithoutCal, LVAL, llrK, labelK,
                  f"GMM - calibration validation priorCal = {priorCal}", "green")


# CALIBRATION FUSION
# def calibrationFusion(scoreQLR, scoreSVM, scoreGMM, LVAL, K, priorT, priorCal, printResult=False):
#     calibratedSVALK = []
#     labelK = []
#
#     for i in range(K):
#         SCAL_QLR, SVAL_QLR = np.hstack([scoreQLR[jdx::K] for jdx in range(K) if jdx != i]), scoreQLR[i::K]
#         SCAL_SVM, SVAL_SVM = np.hstack([scoreSVM[jdx::K] for jdx in range(K) if jdx != i]), scoreSVM[i::K]
#         SCAL_GMM, SVAL_GMM = np.hstack([scoreGMM[jdx::K] for jdx in range(K) if jdx != i]), scoreGMM[i::K]
#
#         SCAL = np.hstack([SCAL_QLR, SCAL_SVM, SCAL_GMM])
#         SVAL = np.hstack([SVAL_QLR, SVAL_SVM, SVAL_GMM])
#
#         labelCal, labelVal = np.hstack([LVAL[jdx::K] for jdx in range(K) if jdx != i]), LVAL[i::K]
#         logRegWeight = priorWeightedLogClass(vrow(SCAL), labelCal, 0, priorCal)
#         vf = \
#             sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
#         w, b = vf[:-1], vf[-1]
#         calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorCal / (1 - priorCal))).ravel()
#         calibratedSVALK.append(calibrated_SVAL)
#         labelK.append(labelVal)
#
#     llrK = np.hstack(calibratedSVALK)
#     labelK = np.hstack(labelK)
#     minDCFKFold = minDCF(llrK, labelK, priorT, 1, 1)
#     actDCFKFold = actDCF(llrK, labelK, priorT, 1, 1)
#
#     if printResult:
#         print(f"\tRESULT FOR CALIBRATION FUSION")
#         print(f"\tminDCF - cal: {minDCFKFold:.4f}")
#         print(f"\tactDCF - cal: {actDCFKFold:.4f}")


# PRINT MAIN INFORMATION AND PLOT GRAPH NOT FOR FUSION
def printData(minDCFWithoutCal, actDCFWithoutCal, minDCFKFold, actDCFKFold, sllrWithoutCal, LVAL, llrK, labelK,
              titleGraph, colorGraph):
    print(f"\t\tminDCF: {minDCFWithoutCal:.4f}")
    print(f"\t\tactDCF: {actDCFWithoutCal:.4f}")
    print(f"\t\tminDCF - cal: {minDCFKFold:.4f}")
    print(f"\t\tactDCF - cal: {actDCFKFold:.4f}")
    effPriorLogOdds, actDCFWithoutCalBayesError, minDCFWithoutCalBayesError = bayesError(
        llr=sllrWithoutCal,
        LTE=LVAL,
        lineLeft=-4,
        lineRight=4)
    createBayesErrorPlots(effPriorLogOdds, actDCFWithoutCalBayesError, minDCFWithoutCalBayesError, [-4, 4],
                          [0, 1], colorGraph, colorGraph, titleGraph,
                          False, "actDCF", "min DCF", "-.",
                          ":")
    effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError = bayesError(
        llr=llrK,
        LTE=labelK,
        lineLeft=-4,
        lineRight=4)
    createBayesErrorPlots(effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError, [-4, 4],
                          [0, 1], colorGraph, colorGraph, titleGraph,
                          True, "actDCF - cal", "min DCF - cal", "-",
                          "--")
