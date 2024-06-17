import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from SVMClass import SVM
from graph import createMinDCFActDCFPlot


def polinomialSVM(DTR, LTR, DVAL, LVAL, printResult=True, titleGraph=""):
    resultPolynomial = {}
    count = 0
    CValues = np.logspace(-5, 0, 11)
    zeta = 0
    K = 0
    c = 1
    d = 2
    priorT = 0.1
    for C in CValues:
        resultPolynomial["config" + str(count)] = {
            "K": K,
            "C": C,
            "c": c
        }
        polinomial = SVM(DTR, LTR, C, K, "polinomial", c=c, d=d)
        alphaStar, _, _ = sp.fmin_l_bfgs_b(func=polinomial.fOpt,
                                           x0=np.zeros(polinomial.getDTRExtend().shape[1]),
                                           approx_grad=False,
                                           maxfun=15000,
                                           factr=1.0,
                                           bounds=[(0, C) for i in LTR],
                                           )

        resultPolynomial["config" + str(count)]["dualLoss"] = -polinomial.fOpt(alphaStar)[0][0]

        sllr = polinomial.computeScore(
            alphaStar=alphaStar,
            D2=DVAL)
        Pval = (sllr > 0) * 1
        resultPolynomial["config" + str(count)]["errorRate"] = (Pval != LVAL).sum() / float(LVAL.size)
        resultPolynomial["config" + str(count)]["minDCF"] = minDCF(sllr, LVAL, priorT, 1, 1)
        resultPolynomial["config" + str(count)]["actDCF"] = actDCF(sllr, LVAL, priorT, 1, 1)

        count += 1

    if printResult:
        x = []
        yMinDCF = []
        yActDCF = []
        print("RESULT POLYNOMIAL SVM")
        for key in resultPolynomial.keys():
            print(
                f"\t{key} where K={resultPolynomial[key]["K"]} and C = {resultPolynomial[key]["C"]}, Poly(d = {d}, c = {resultPolynomial[key]["c"]})")
            print(f"\t\tDual Loss: {resultPolynomial[key]["dualLoss"]:.6e}")
            print(f"\t\tError Rate: {resultPolynomial[key]["errorRate"] * 100:.1f}%")
            print(f"\t\tminDCF: {resultPolynomial[key]["minDCF"]:.4f}")
            print(f"\t\tactDCF: {resultPolynomial[key]["actDCF"]:.4f}")
            print()
            x.append(resultPolynomial[key]["C"])
            yMinDCF.append(resultPolynomial[key]["minDCF"])
            yActDCF.append(resultPolynomial[key]["actDCF"])

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
