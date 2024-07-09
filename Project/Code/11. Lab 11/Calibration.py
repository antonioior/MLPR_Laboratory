import numpy as np
import scipy.optimize as sp
from SVM import SVM

from DCF import minDCF, actDCF
from QuadraticLogisticRegression import quadraticLogClass
from utils import vrow


def calibration(DTR, LTR, DVAL, LVAL):
    K = 5
    # calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, 0.1)
    calibrationSVM()


# TODO: Implement calibrationLogisticRegression
def calibrationLogisticRegression(DTR, LTR, DVAL, LVAL, K, priorT, printResult=False):
    l = 3.162278e-4
    calibratedSVALK = []
    labelK = []

    for i in range(K):
        SCAL, SVAL = np.hstack([DTR[jdx::K] for jdx in range(K) if jdx != i]), DTR[i::K]
        LCAL, LVAL = np.hstack([LTR[jdx::K] for jdx in range(K) if jdx != i]), LTR[i::K]
        logRegObj = quadraticLogClass(SCAL, LCAL, l)
        vf = sp.fmin_l_bfgs_b(func=logRegObj.logreg_obj, x0=np.zeros(SCAL.shape[0] + 1))[0]
        w, b = vf[:-1], vf[-1]
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorT / (1 - priorT))).ravel()
        calibratedSVALK.append(calibrated_SVAL)
        labelK.append(LVAL)

    calibratedSVALK = np.hstack(calibratedSVALK)
    labelK = np.hstack(labelK)
    minDCFKFold = minDCF(calibratedSVALK, labelK, priorT, 1, 1)
    actDCFKFold = actDCF(calibratedSVALK, labelK, priorT, 1, 1)

    # w, b = training(vrow(self.scoreCalDat), self.labelsCalDat, 0, self.priorT)
    # self.calibratedKEval = (w.T @ vrow(self.scoreEvalDat) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
    # self.minDCFKFoldEvalCal = minDCF(self.calibratedKEval, self.labelsEvalDat, self.priorT, 1, 1)
    # self.actDCFKFoldEvalCal = actDCF(self.calibratedKEval, self.labelsEvalDat, self.priorT, 1, 1)
    if printResult:
        print("RESULT FOR CALIBRATION LOGISTIC REGRESSION")
        print(f"\tLambda: {l}")
        print(f"\t\tminDCF: {minDCFKFold:.4f}")
        print(f"\t\tactDCF: {actDCFKFold:.4f}")


def calibrationSVM(DTR, LTR, DVAL, LVAL, K, priorT, printResult=False):
    C = 100
    gamma = 0.1
    KRadial = 1
    for i in range(K):
        SCAL, SVAL = np.hstack([DTR[jdx::K] for jdx in range(K) if jdx != i]), DTR[i::K]
        LCAL, LVAL = np.hstack([LTR[jdx::K] for jdx in range(K) if jdx != i]), LTR[i::K]
        radial = SVM(SCAL, LCAL, C, KRadial, "radial", gamma=gamma)
        alphaStar = sp.fmin_l_bfgs_b(func=radial.fOpt,
                                     x0=np.zeros(radial.getDTRExtend().shape[1]),
                                     approx_grad=False,
                                     maxfun=15000,
                                     factr=1.0,
                                     bounds=[(0, C) for i in LTR],
                                     )[0]

        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorT / (1 - priorT))).ravel()
        calibratedSVALK.append(calibrated_SVAL)
        labelK.append(LVAL)

    vf = sp.fmin_l_bfgs_b(func=radial.fOpt,
                          x0=np.zeros(radial.getDTRExtend().shape[1]),
                          approx_grad=False,
                          maxfun=15000,
                          factr=1.0,
                          bounds=[(0, C) for i in LTR],
                          )[0]
    w, b = vf[:-1], vf[-1]
