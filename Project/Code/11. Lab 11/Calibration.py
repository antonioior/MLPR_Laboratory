import numpy as np
import scipy.optimize as sp

from QuadraticLogisticRegression import quadraticLogClass
from DCF import bayesError, minDCF, actDCF
from graph import createBayesErrorPlots
from utils import vrow, logpdf_GMM
from SVMClass import SVM
from LBG import trainGMMReturnMinAndActDCF, trainGMMCalibrationReturnMinAndActDCF


def calibration(DTR, LTR, DVAL, LVAL, printResult=False):
    K = 5
    priorT = 0.1
    priorCals = [0.1, 0.5, 0.9]
    for priorCal in priorCals:
        calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult)
        calibrationSVM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult)
        calibrationGMM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult)


# CALIBRATION LOGISTIC REGRESSION
def calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult):
    l = 3.162278e-2
    logRegQDT = quadraticLogClass(DTR, LTR, l)
    sllrWithoutCal, minDCFWithoutCal, actDCFWithoutCal, score = logRegQDT.trainReturnMinAndActDCF(DVAL, LVAL, priorT)
    minDCFKFold, actDCFKFold, llrK, labelK = logRegQDT.trainCalibrationReturnMinAndActDCF(K, priorCal, priorT, score,
                                                                                          sllrWithoutCal, LVAL)
    if printResult:
        print(f"priorCal {priorCal}")
        print("\tCALIBRATION QUADRATIC LOGISTIC REGRESSION")
        print(f"\tLambda: {l}")
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
                              [0, 1], "b", "b", f"QLR - calibration validation priorCal = {priorCal}",
                              False, "actDCF", "min DCF", "-.",
                              ":")
        effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError = bayesError(
            llr=llrK,
            LTE=labelK,
            lineLeft=-4,
            lineRight=4)
        createBayesErrorPlots(effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError, [-4, 4],
                              [0, 1], "b", "b", f"QLR - calibration validation priorCal = {priorCal}",
                              True, "actDCF - cal", "min DCF - cal", "-",
                              "--")


# CALIBRATION SVM
def calibrationSVM(DTR, LTR, DVAL, LVAL, K, priorT, priorCal, printResult=False):
    C = 100
    gamma = 0.1
    KRadial = 1
    radialSVM = SVM(DTR, LTR, C, KRadial, "radial", gamma=gamma)
    sllrWithoutCal, minDCFWithoutCal, actDCFWithoutCal, score = radialSVM.tainRadialReturnMinAndActDCF(DVAL, LVAL,
                                                                                                       priorT)
    minDCFKFold, actDCFKFold, llrK, labelK = radialSVM.trainCalibrationReturnMinAndActDCF(K, priorCal, priorT, score,
                                                                                          sllrWithoutCal, LVAL)

    if printResult:
        print("\tRESULT FOR CALIBRATION SVM")
        print(f"\tC: {C}, \u03B3: {gamma}, KRadial: {KRadial}")
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
                              [0, 1], "orange", "orange", f"SVM - calibration validation priorCal = {priorCal}",
                              False, "actDCF", "min DCF", "-.",
                              ":")
        effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError = bayesError(
            llr=llrK,
            LTE=labelK,
            lineLeft=-4,
            lineRight=4)
        createBayesErrorPlots(effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError, [-4, 4],
                              [0, 1], "orange", "orange", f"SVM - calibration validation priorCal = {priorCal}",
                              True, "actDCF - cal", "min DCF - cal", "-",
                              "--")


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
        print("\tRESULT FOR CALIBRATION GMM")
        print(
            f"\tcovType: {covType}, component0 = {componentGMM0}, component1 = {componentGMM1}, psi: {psi}, alpha: {alpha}")
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
                              [0, 1], "green", "green", f"GMM - calibration validation priorCal = {priorCal}",
                              False, "actDCF", "min DCF", "-.",
                              ":")
        effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError = bayesError(
            llr=llrK,
            LTE=labelK,
            lineLeft=-4,
            lineRight=4)
        createBayesErrorPlots(effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError, [-4, 4],
                              [0, 1], "green", "green", f"GMM - calibration validation priorCal = {priorCal}",
                              True, "actDCF - cal", "min DCF - cal", "-",
                              "--")
