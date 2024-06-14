import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from utils import vrow, vcol, errorRate


def SVM(DTR, LTR, DVAL, LVAL, printResult=False):
    result = {}
    count = 0
    for K in [1, 10]:
        for C in [0.1, 1, 10]:
            result["config" + str(count)] = {
                "K": K,
                "C": C
            }
            dualSVMObj = dualSVM(DTR, LTR, C, K)
            alphaStar, _, _ = sp.fmin_l_bfgs_b(func=dualSVMObj.fOpt,
                                               x0=np.zeros(dualSVMObj.getDTRExtend().shape[1]),
                                               approx_grad=False,
                                               maxfun=15000,
                                               factr=1.0,
                                               bounds=[(0, C) for i in LTR],
                                               )
            WHat = computeWHat(alphaStar, dualSVMObj.getZTR(), dualSVMObj.getDTRExtend())
            result["config" + str(count)]["primalLoss"] = dualSVMObj.primalLoss(WHat)
            result["config" + str(count)]["dualLoss"] = -dualSVMObj.fOpt(alphaStar)[0][0]
            result["config" + str(count)]["dualityGap"] = result["config" + str(count)]["primalLoss"] - \
                                                          result["config" + str(count)]["dualLoss"]
            result["config" + str(count)]["errorRate"], sVal = errorRate(DVAL, LVAL, WHat, K)

            pEmp = (LTR == 1).sum() / LTR.size
            sllr = sVal - np.log(pEmp / (1 - pEmp))
            result["config" + str(count)]["minDCF"] = minDCF(sllr, LVAL, 0.5, 1, 1)
            result["config" + str(count)]["actDCF"] = actDCF(sllr, LVAL, 0.5, 1, 1)
            count += 1

    if printResult:
        print("RESULT DUAL SVM")
        for key in result.keys():
            print(f"\t{key} where k={result[key]["K"]} and C = {result[key]["C"]}")
            print(f"\t\tPrimal Loss: {result[key]["primalLoss"]:.6e}")
            print(f"\t\tDual Loss: {result[key]["dualLoss"]:.6e}")
            print(f"\t\tDuality Gap: {result[key]["dualityGap"]:.6e}")
            print(f"\t\tError Rate: {result[key]["errorRate"] * 100 :.1f}%")
            print(f"\t\tminDCF: {result[key]["minDCF"]:.4f}")
            print(f"\t\tactDCF: {result[key]["actDCF"]:.4f}")
            print()


class dualSVM:

    def __init__(self, DTR, LTR, C, K):
        self.DTRExtend = computeDhat(DTR, K)
        self.ZTR = LTR * 2. - 1.
        self.C = C
        self.K = K
        self.H = computeHhat(computeGhat(self.DTRExtend), self.ZTR)

    def fOpt(self, alpha):
        Ha = self.H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    def getDTRExtend(self):
        return self.DTRExtend

    def getZTR(self):
        return self.ZTR

    def primalLoss(self, WHat):
        S = (vrow(WHat) @ self.DTRExtend).ravel()
        return 0.5 * np.linalg.norm(WHat) ** 2 + self.C * np.maximum(0, 1 - self.ZTR * S).sum()


def computeDhat(D, K):
    newRow = np.full((1, D.shape[1]), K)
    return np.concatenate((D, newRow), axis=0)


def computeGhat(Dhat):
    return np.dot(Dhat.T, Dhat)


def computeHhat(Ghat, z):
    return Ghat * vcol(z) * vrow(z)


def computeWHat(alpha, ZTR, DTR_EXT):
    return (vrow(alpha) * vrow(ZTR) * DTR_EXT).sum(axis=1)
