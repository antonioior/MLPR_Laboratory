import numpy as np
import scipy.optimize as sp

from EMGMM import EMGmm, smoothCovariance
from utils import compute_mu_C, logpdf_GMM
from DCF import minDCF, actDCF
from PriorWeightedBinLogReg import priorWeightedLogClass
from utils import vrow


def LBGAlgorithm(X, alpha, numComponent, psi=None, covType="full"):
    mu, C = compute_mu_C(X)

    if covType == "diagonal":
        C = C * np.eye(X.shape[0])

    if psi is not None:
        gmm = [(1.0, mu, smoothCovariance(C, psi))]
    else:
        gmm = [(1.0, mu, C)]

    while len(gmm) < numComponent:
        gmm = split_GMM(gmm, alpha)
        _, gmm = EMGmm(X, gmm, psi=psi, covType=covType)
    return gmm


def split_GMM(gmm, alpha):
    gmmOut = []
    for gIndex in range(len(gmm)):
        w, mu, C = gmm[gIndex]
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmOut.append((w * 0.5, mu - d, C))
        gmmOut.append((w * 0.5, mu + d, C))
    return gmmOut


# LAB 11
def trainGMMReturnMinAndActDCF(DTR, LTR, DVAL, LVAL, priorT, alpha, componentGMM0, componentGMM1, psi, covType):
    gmm0 = LBGAlgorithm(DTR[:, LTR == 0], alpha, componentGMM0, psi=psi, covType=covType)
    gmm1 = LBGAlgorithm(DTR[:, LTR == 1], alpha, componentGMM1, psi=psi, covType=covType)
    sllr = logpdf_GMM(DVAL, gmm1)[1] - logpdf_GMM(DVAL, gmm0)[1]
    minDCFWithoutCal = minDCF(sllr, LVAL, priorT, 1.0, 1.0)
    actDCFWithoutCal = actDCF(sllr, LVAL, priorT, 1.0, 1.0)
    return sllr, minDCFWithoutCal, actDCFWithoutCal


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
