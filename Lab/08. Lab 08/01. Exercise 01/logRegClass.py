import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from utils import vcol, vrow, errorRate


def binaryLogisticRegression(DTR, LTR, DVAL, LVAL, printResult=False):
    result = {
        1e-3: {},
        1e-1: {},
        1.: {}
    }

    for lamb in result.keys():
        logRegObj = logRegClass(DTR, LTR, lamb)
        vf = sp.fmin_l_bfgs_b(func=logRegObj.logreg_obj, x0=np.zeros(DTR.shape[0] + 1),
                              approx_grad=False, maxfun=15000)[0]
        result[lamb]["J"] = logRegObj.logreg_obj(vf)[0]
        result[lamb]["ErrorRate"], sVal = errorRate(DVAL, LVAL, vf)
        # Empirical Prior
        pEmp = (LTR == 1).sum() / LTR.size
        sllr = sVal - np.log(pEmp / (1 - pEmp))
        result[lamb]["minDCF"] = minDCF(sllr, LVAL, 0.5, 1, 1)
        result[lamb]["actDCF"] = actDCF(sllr, LVAL, 0.5, 1, 1)

    if printResult:
        print("RESULT FOR BINARY LOGISTIC REGRESSION")
        for lamb in result.keys():
            print(f"\tLambda: {lamb}")
            print(f"\t\tJ: {result[lamb]["J"]: .6e}")
            print(f"\t\tError: {result[lamb]["ErrorRate"] * 100:.1f} %")
            print(f"\t\tminDCF: {result[lamb]['minDCF']:.4f}")
            print(f"\t\tactDCF: {result[lamb]['actDCF']:.4f}")
            print()


class logRegClass:

    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l

    def logreg_obj(self, v):
        w, b = v[:-1], v[-1]
        S = np.dot(vcol(w).T, self.DTR).ravel() + b

        regressionTerm = self.l / 2 * np.linalg.norm(w) ** 2

        ZTR = 2. * self.LTR - 1.
        loss = np.logaddexp(0, -S * ZTR)
        J = regressionTerm + loss.mean()

        G = -ZTR / (1 + np.exp(ZTR * S))

        gradB = G.mean()
        gradW = self.l * w.ravel() + (vrow(G) * self.DTR).mean(1)
        grad = np.hstack([gradW, np.array(gradB)])
        return J, grad
