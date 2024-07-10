import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from PriorWeightedBinLogReg import priorWeightedLogClass
from QuadraticLogisticRegression import quadraticLogClass, expandeFeature
from utils import errorRate, vrow


def calibration(DTR, LTR, DVAL, LVAL, printResult=False):
    K = 5
    calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, 0.1, printResult)
    # calibrationSVM()


def calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, printResult):
    l = 3.162278e-4
    calibratedSVALK = []
    labelK = []

    logRegQDT = quadraticLogClass(DTR, LTR, l)
    vf = sp.fmin_l_bfgs_b(func=logRegQDT.logreg_obj, x0=np.zeros(logRegQDT.DTR.shape[0] + 1))[0]
    DVAL_expanded = expandeFeature(DVAL)
    _, score = errorRate(DVAL_expanded, LVAL, vf)
    pEmp = (LTR == 1).sum() / LTR.size
    sllr = score - np.log(pEmp / (1 - pEmp))
    minDCFWithoutCal = minDCF(sllr, LVAL, priorT, 1, 1)
    actDCFWithoutCal = actDCF(sllr, LVAL, priorT, 1, 1)

    for i in range(K):
        SCAL, SVAL = np.hstack([score[jdx::K] for jdx in range(K) if jdx != i]), score[i::K]
        labelCal, labelVal = np.hstack([LVAL[jdx::K] for jdx in range(K) if jdx != i]), LVAL[i::K]
        w, b = training(vrow(SCAL), labelCal, 0, priorT)
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorT / (1 - priorT))).ravel()
        calibratedSVALK.append(calibrated_SVAL)
        labelK.append(LVAL)

    calibratedSVALK = np.hstack(calibratedSVALK)
    labelK = np.hstack(labelK)
    minDCFKFold = minDCF(calibratedSVALK, labelK, priorT, 1, 1)
    actDCFKFold = actDCF(calibratedSVALK, labelK, priorT, 1, 1)

    if printResult:
        print("RESULT FOR CALIBRATION LOGISTIC REGRESSION")
        print(f"\tLambda: {l}")
        print(f"\t\tminDCF: {minDCFWithoutCal:.4f}")
        print(f"\t\tactDCF: {actDCFWithoutCal:.4f}")
        print(f"\t\tminDCF: {minDCFKFold:.4f}")
        print(f"\t\tactDCF: {actDCFKFold:.4f}")


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


def training(SCAL, LCAL, l, priorT):
    logRegWeight = priorWeightedLogClass(SCAL, LCAL, l, priorT)
    vf = sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
    return vf[:-1], vf[-1]
