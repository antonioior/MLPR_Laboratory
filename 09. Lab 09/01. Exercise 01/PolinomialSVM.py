import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from SVMClass import SVM


def polinomialSVM(DTR, LTR, DVAL, LVAL, printResult=True):
    resultPolynomial = {}
    count = 0
    C = 1
    for c in [0, 1]:
        for K in [0, 1]:
            resultPolynomial["config" + str(count)] = {
                "K": K,
                "C": C,
                "c": c
            }
            polinomial = SVM(DTR, LTR, C, K, "polinomial", c)
            alphaStar, _, _ = sp.fmin_l_bfgs_b(func=polinomial.fOpt,
                                               x0=np.zeros(polinomial.getDTRExtend().shape[1]),
                                               approx_grad=False,
                                               maxfun=15000,
                                               factr=1.0,
                                               bounds=[(0, C) for i in LTR],
                                               )

            resultPolynomial["config" + str(count)]["dualLoss"] = -polinomial.fOpt(alphaStar)[0][0]

            sllr = polinomial.computeScore(
                sVal=None,
                pEmp=None,
                alphaStar=alphaStar,
                D2=DVAL)
            Pval = (sllr > 0) * 1
            resultPolynomial["config" + str(count)]["errorRate"] = (Pval != LVAL).sum() / float(LVAL.size)
            resultPolynomial["config" + str(count)]["minDCF"] = minDCF(sllr, LVAL, 0.5, 1, 1)
            resultPolynomial["config" + str(count)]["actDCF"] = actDCF(sllr, LVAL, 0.5, 1, 1)

            count += 1

    if printResult:
        print("RESULT POLYNOMIAL SVM")
        for key in resultPolynomial.keys():
            print(
                f"\t{key} where K={resultPolynomial[key]["K"]} and C = {resultPolynomial[key]["C"]}, Poly(d = {2}, c = {resultPolynomial[key]["c"]})")
            print(f"\t\tDual Loss: {resultPolynomial[key]["dualLoss"]:.6e}")
            print(f"\t\tError Rate: {resultPolynomial[key]["errorRate"] * 100:.1f}%")
            print(f"\t\tminDCF: {resultPolynomial[key]["minDCF"]:.4f}")
            print(f"\t\tactDCF: {resultPolynomial[key]["actDCF"]:.4f}")
            print()
