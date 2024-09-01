import numpy as np
import scipy.optimize as sp

from EMGMM import EMGmm
from utils import logpdf_GMM
from DCF import minDCF, actDCF
from PriorWeightedBinLogReg import priorWeightedLogClass
from utils import vrow


def LBGAlgorithm(X, alpha, numComponent, psi=None, covType="full"):
    mean = X.mean(axis=1).reshape(-1, 1)
    cov = 1 / X.shape[1] * np.dot(X - mean, (X - mean).T)
    gmm = [(1.0, mean, cov)]

    while len(gmm) <= numComponent:
        gmm, _ = EMGmm(X, gmm, psi=psi, covType=covType)
        if len(gmm) == numComponent:
            break

        newGmm = []
        for i in range(len(gmm)):
            (w, mu, sigma) = gmm[i]
            U, s, Vh = np.linalg.svd(sigma)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha

            newGmm.append((w / 2, mu - d, sigma))
            newGmm.append((w / 2, mu + d, sigma))
        gmm = newGmm
    return gmm


# LAB 11
def trainGMMCalibrationReturnMinAndActDCF(K, priorCal, priorT, sllrWithoutCal, LVAL):
    calibratedSVALK = []
    labelK = []

    for i in range(K):
        SCAL, SVAL = np.hstack([sllrWithoutCal[jdx::K] for jdx in range(K) if jdx != i]), sllrWithoutCal[i::K]
        labelCal, labelVal = np.hstack([LVAL[jdx::K] for jdx in range(K) if jdx != i]), LVAL[i::K]
        logRegWeight = priorWeightedLogClass(vrow(SCAL), labelCal, 0, priorCal)
        vf = \
            sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
        w, b = vf[:-1], vf[-1]
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(priorCal / (1 - priorCal))).ravel()
        calibratedSVALK.append(calibrated_SVAL)
        labelK.append(labelVal)

    llrK = np.hstack(calibratedSVALK)
    labelK = np.hstack(labelK)
    minDCFKFold = minDCF(llrK, labelK, priorT, 1, 1)
    actDCFKFold = actDCF(llrK, labelK, priorT, 1, 1)
    return minDCFKFold, actDCFKFold, llrK, labelK
