import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from utils import vcol, vrow


def priorWeightedLogisticRegression(DTR, LTR, DVAL, LVAL, printResult=False):
    result = {
        1e-3: {},
        1e-1: {},
        1.: {}
    }
    pT = 0.8
    for lamb in result.keys():
        logRegObj = priorWeightedLogClass(DTR, LTR, lamb, pT)
        vf = sp.fmin_l_bfgs_b(func=logRegObj.logreg_obj, x0=np.zeros(DTR.shape[0] + 1),
                              approx_grad=False, maxfun=15000)[0]
        result[lamb]["J"] = logRegObj.logreg_obj(vf)[0]

        sVal = np.dot(vf[:-1].T, DVAL) + vf[-1]
        sllr = sVal - np.log(pT / (1 - pT))
        result[lamb]["minDCF"] = minDCF(sllr, LVAL, pT, 1, 1)
        result[lamb]["actDCF"] = actDCF(sllr, LVAL, pT, 1, 1)

    if printResult:
        print("RESULT FOR PRIOR-WEIGHTED LOGISTIC REGRESSION")
        for lamb in result.keys():
            print(f"\tLambda: {lamb}")
            print(f"\t\tJ: {result[lamb]["J"]: .6e}")
            print(f"\t\tminDCF: {result[lamb]['minDCF']:.4f}")
            print(f"\t\tactDCF: {result[lamb]['actDCF']:.4f}")
            print()


class priorWeightedLogClass:
    def __init__(self, DTR, LTR, l, pT):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pT = pT

    def logreg_obj(self, v):
        w, b = v[:-1], v[-1]
        S = np.dot(vcol(w).T, self.DTR).ravel() + b
        ZTR = 2. * self.LTR - 1.

        wTrue = self.pT / (ZTR > 0).sum()
        wFalse = (1 - self.pT) / (ZTR < 0).sum()

        regressionTerm = self.l / 2 * np.linalg.norm(w) ** 2

        loss = np.logaddexp(0, -S * ZTR)
        loss[ZTR > 0] *= wTrue
        loss[ZTR < 0] *= wFalse

        J = regressionTerm + loss.sum()

        G = -ZTR / (1 + np.exp(ZTR * S))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse

        gradB = G.sum()
        gradW = self.l * w.ravel() + (vrow(G) * self.DTR).sum(1)
        grad = np.hstack([gradW, np.array(gradB)])
        return J, grad
