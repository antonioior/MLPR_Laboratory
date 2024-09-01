import numpy as np

from utils import vrow, vcol
import scipy.optimize as sp
from PriorWeightedBinLogReg import priorWeightedLogClass
from DCF import minDCF, actDCF


# SVM class
class SVM:

    def __init__(self, DTR, LTR, C, K, SVMType="linear", c=None, gamma=None, d=None):
        self.SVMType = SVMType
        self.DTR = DTR
        self.K = K
        self.LTR = LTR
        self.ZTR = LTR * 2. - 1.
        self.C = C
        self.kernel = None

        self.DTRHat = self.computeDhat()
        self.GHat = self.computeGhat()

        if self.SVMType == "polinomial":
            self.kernel = polinomialKernel(c, d, K ** 2)
        elif self.SVMType == "radial":
            self.kernel = radialKernel(gamma, K ** 2)
        self.H = self.computeHhat()

    def fOpt(self, alpha):
        Ha = self.H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    def getDTRExtend(self):
        return self.DTRHat

    def primalLoss(self, WHat):
        S = (vrow(WHat) @ self.DTRHat).ravel()
        return 0.5 * np.linalg.norm(WHat) ** 2 + self.C * np.maximum(0, 1 - self.ZTR * S).sum()

    def computeDhat(self):
        newRow = np.full((1, self.DTR.shape[1]), self.K)
        return np.concatenate((self.DTR, newRow), axis=0)

    def computeGhat(self):
        return np.dot(self.DTRHat.T, self.DTRHat)

    def computeHhat(self):
        if self.SVMType == "linear":
            return self.GHat * vcol(self.ZTR) * vrow(self.ZTR)
        else:
            return self.kernel.k(self.DTR, self.DTR) * vcol(self.ZTR) * vrow(self.ZTR)

    def computeWHat(self, alpha):
        return (vrow(alpha) * vrow(self.ZTR) * self.DTRHat).sum(axis=1)

    def computeScore(self, alphaStar, D2):
        return (vcol(alphaStar) * vcol(self.ZTR) * self.kernel.k(self.DTR, D2)).sum(0)

    # LAB 11
    def tainRadialReturnMinAndActDCF(self, DVAL, LVAL, priorT):
        from DCF import minDCF, actDCF

        self.alphaStar, _, _ = sp.fmin_l_bfgs_b(func=self.fOpt,
                                                x0=np.zeros(self.getDTRExtend().shape[1]),
                                                approx_grad=False,
                                                maxfun=15000,
                                                factr=1.0,
                                                bounds=[(0, self.C) for i in self.LTR],
                                                )
        sllr = self.computeScore(
            alphaStar=self.alphaStar,
            D2=DVAL)
        minDCF = minDCF(sllr, LVAL, priorT, 1, 1)
        actDCF = actDCF(sllr, LVAL, priorT, 1, 1)
        return sllr, minDCF, actDCF

    def trainCalibrationReturnMinAndActDCF(self, K, priorCal, priorT, score, LVAL):
        calibratedSVALK = []
        labelK = []

        for i in range(K):
            SCAL, SVAL = np.hstack([score[jdx::K] for jdx in range(K) if jdx != i]), score[i::K]
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


# Kernel classes
# POLINOMIAL KERNEL
class polinomialKernel:

    def __init__(self, c, d, zeta):
        self.c = c
        self.d = d
        self.zeta = zeta

    def k(self, D1, D2):
        return ((np.dot(D1.T, D2) + self.c) ** self.d) + self.zeta


# RADIAL KERNEL
class radialKernel:

    def __init__(self, gamma, zeta):
        self.gamma = gamma
        self.zeta = zeta

    def k(self, D1, D2):
        D1Norms = (D1 ** 2).sum(0)
        D2Norms = (D2 ** 2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return (np.exp(-self.gamma * Z)) + self.zeta
