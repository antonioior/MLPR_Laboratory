import numpy as np
import scipy.optimize as sp

from PriorWeightedBinLogReg import priorWeightedLogClass
from QuadraticLogisticRegression import quadraticLogClass
from DCF import bayesError, minDCF, actDCF
from graph import createBayesErrorPlots
from utils import vrow


def calibration(DTR, LTR, DVAL, LVAL, printResult=False):
    K = 5
    calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, 0.1, printResult)
    # calibrationSVM()


def calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, printResult):
    l = 3.162278e-2
    calibratedSVALK = []
    labelK = []

    logRegQDT = quadraticLogClass(DTR, LTR, l)
    sllrWithoutCal, minDCFWithoutCal, actDCFWithoutCal, score = logRegQDT.trainReturnMinAndActDCF(DVAL, LVAL, priorT)

    for i in range(K):
        SCAL, SVAL = np.hstack([score[jdx::K] for jdx in range(K) if jdx != i]), sllrWithoutCal[i::K]
        labelCal, labelVal = np.hstack([LVAL[jdx::K] for jdx in range(K) if jdx != i]), LVAL[i::K]
        logRegWeight = priorWeightedLogClass(vrow(SCAL), labelCal, 0, priorT)
        vf = \
            sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
        w, b = vf[:-1], vf[-1]
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorT / (1 - priorT))).ravel()
        calibratedSVALK.append(calibrated_SVAL)
        labelK.append(labelVal)

    llrK = np.hstack(calibratedSVALK)
    labelK = np.hstack(labelK)
    minDCFKFold = minDCF(llrK, labelK, priorT, 1, 1)
    actDCFKFold = actDCF(llrK, labelK, priorT, 1, 1)

    if printResult:
        print("RESULT FOR CALIBRATION LOGISTIC REGRESSION")
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
                              [0, 1], "b", "b", "QLR - calibration validation",
                              False, "actDCF", "min DCF", "-.",
                              ":")
        effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError = bayesError(
            llr=llrK,
            LTE=labelK,
            lineLeft=-4,
            lineRight=4)
        createBayesErrorPlots(effPriorLogOdds, actDCFKFoldBayesError, minDCFKFoldBayesError, [-4, 4],
                              [0, 1], "b", "b", "QLR - calibration validation",
                              True, "actDCF - cal", "min DCF - cal", "-",
                              "--")

# def calibrationSVM(DTR, LTR, DVAL, LVAL, K, priorT, printResult=False):
#     C = 100
#     gamma = 0.1
#     KRadial = 1
#     for i in range(K):
#         SCAL, SVAL = np.hstack([DTR[jdx::K] for jdx in range(K) if jdx != i]), DTR[i::K]
#         LCAL, LVAL = np.hstack([LTR[jdx::K] for jdx in range(K) if jdx != i]), LTR[i::K]
#         radial = SVM(SCAL, LCAL, C, KRadial, "radial", gamma=gamma)
#         alphaStar = sp.fmin_l_bfgs_b(func=radial.fOpt,
#                                      x0=np.zeros(radial.getDTRExtend().shape[1]),
#                                      approx_grad=False,
#                                      maxfun=15000,
#                                      factr=1.0,
#                                      bounds=[(0, C) for i in LTR],
#                                      )[0]
#
#         calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorT / (1 - priorT))).ravel()
#         calibratedSVALK.append(calibrated_SVAL)
#         labelK.append(LVAL)
#
#     vf = sp.fmin_l_bfgs_b(func=radial.fOpt,
#                           x0=np.zeros(radial.getDTRExtend().shape[1]),
#                           approx_grad=False,
#                           maxfun=15000,
#                           factr=1.0,
#                           bounds=[(0, C) for i in LTR],
#                           )[0]
#     w, b = vf[:-1], vf[-1]
