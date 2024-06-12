import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from graph import createMinDCFActDCFPlot
from utils import vcol, vrow


def PriorWeightedBinLogReg(DTR, LTR, DVAL, LVAL, printResult=False):
    lamb = np.logspace(-4, 2, 13)
    result = {}
    pT = 0.1
    for i in range(len(lamb)):
        result["lamb" + str(i)] = {"lambda": lamb[i]}

    for i in range(len(lamb)):
        logRegObj = priorWeightedLogClass(DTR, LTR, result["lamb" + str(i)]["lambda"], pT)
        vf = sp.fmin_l_bfgs_b(func=logRegObj.logreg_obj, x0=np.zeros(DTR.shape[0] + 1),
                              approx_grad=False, maxfun=15000)[0]
        result["lamb" + str(i)]["J"] = logRegObj.logreg_obj(vf)[0]

        sVal = np.dot(vf[:-1].T, DVAL) + vf[-1]
        sllr = sVal - np.log(pT / (1 - pT))
        result["lamb" + str(i)]["minDCF"] = minDCF(sllr, LVAL, pT, 1, 1)
        result["lamb" + str(i)]["actDCF"] = actDCF(sllr, LVAL, pT, 1, 1)

    if printResult:
        x = []
        yMinDCF = []
        yActDCF = []
        print("RESULT FOR PRIOR-WEIGHTED LOGISTIC REGRESSION")
        for lamb in result.keys():
            print(f"\tLambda: {lamb}")
            print(f"\t\tJ: {result[lamb]["J"]: .6e}")
            print(f"\t\tminDCF: {result[lamb]['minDCF']:.4f}")
            print(f"\t\tactDCF: {result[lamb]['actDCF']:.4f}")
            print()
            x.append(result[lamb]["lambda"])
            yMinDCF.append(result[lamb]["minDCF"])
            yActDCF.append(result[lamb]["actDCF"])

        createMinDCFActDCFPlot(
            x=x,
            yMinDCF=yMinDCF,
            yActDCF=yActDCF,
            colorMinDCF="blue",
            colorActDCF="red",
            title="Binary Logistic Regression with prior weighted",
            togheter=True,
            logScale=True
        )

        createMinDCFActDCFPlot(
            x=x,
            yMinDCF=yMinDCF,
            yActDCF=yActDCF,
            colorMinDCF="blue",
            colorActDCF="red",
            title="Binary Logistic Regression with prior weighted",
            togheter=False,
            logScale=True
        )


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
