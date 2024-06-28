import numpy as np

from utils import vrow, vcol


# SVM class
class SVM:

    def __init__(self, DTR, LTR, C, K, SVMType="linear", c=None, gamma=None):
        self.SVMType = SVMType
        self.DTR = DTR
        self.K = K
        self.ZTR = LTR * 2. - 1.
        self.C = C
        self.kernel = None

        self.DTRHat = self.computeDhat()
        self.GHat = self.computeGhat()

        if self.SVMType == "polinomial":
            self.kernel = polinomialKernel(c, 2, K ** 2)
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

    def computeScore(self, sVal, pEmp, alphaStar, D2=None):
        if self.SVMType == "linear":
            return sVal - np.log(pEmp / (1 - pEmp))
        else:
            return (vcol(alphaStar) * vcol(self.ZTR) * self.kernel.k(self.DTR, D2)).sum(0)


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
