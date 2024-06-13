import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from graph import createMinDCFActDCFPlot
from utils import vcol, vrow, errorRate


def QuadraticLogisticRegression(DTR, LTR, DVAL, LVAL, titleGraph, printResult=False):
    lamb = np.logspace(-4, 2, 13)
    result = {}

    for i in range(len(lamb)):
        result["lamb" + str(i)] = {"lambda": lamb[i]}

    for i in range(len(lamb)):
        logRegObj = quadraticLogClass(DTR, LTR, result["lamb" + str(i)]["lambda"])
        vf = sp.fmin_l_bfgs_b(func=logRegObj.logreg_obj, x0=np.zeros(DTR.shape[0] * 2 + 1),
                              approx_grad=False, maxfun=15000)[0]
        result["lamb" + str(i)]["J"] = logRegObj.logreg_obj(vf)[0]
        DVAL_expanded = np.concatenate([DVAL, DVAL ** 2], axis=0)
        result["lamb" + str(i)]["ErrorRate"], sVal = errorRate(DVAL_expanded, LVAL, vf)

        pEmp = (LTR == 1).sum() / LTR.size
        sllr = sVal - np.log(pEmp / (1 - pEmp))
        result["lamb" + str(i)]["minDCF"] = minDCF(sllr, LVAL, 0.1, 1, 1)
        result["lamb" + str(i)]["actDCF"] = actDCF(sllr, LVAL, 0.1, 1, 1)

    if printResult:
        x = []
        yMinDCF = []
        yActDCF = []
        print("RESULT FOR BINARY LOGISTIC REGRESSION")
        for lamb in result.keys():
            print(f"\tLambda: {lamb}")
            print(f"\t\tJ: {result[lamb]["J"]: .6e}")
            print(f"\t\tError: {result[lamb]["ErrorRate"] * 100:.1f} %")
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
            title=titleGraph,
            togheter=True,
            logScale=True
        )

        createMinDCFActDCFPlot(
            x=x,
            yMinDCF=yMinDCF,
            yActDCF=yActDCF,
            colorMinDCF="blue",
            colorActDCF="red",
            title=titleGraph,
            togheter=False,
            logScale=True
        )


class quadraticLogClass:

    def __init__(self, DTR, LTR, l):
        self.DTR = np.concatenate([DTR, DTR ** 2], axis=0)
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
