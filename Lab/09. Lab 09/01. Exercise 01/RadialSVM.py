import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from SVMClass import SVM


def radialSVM(DTR, LTR, DVAL, LVAL, printResult=True):
    resultRadial = {}
    count = 0
    C = 1
    for gamma in [1, 10]:
        for K in [0, 1]:
            resultRadial["config" + str(count)] = {
                "K": K,
                "C": C,
                "gamma": gamma
            }
            radial = SVM(DTR, LTR, C, K, "radial", gamma=gamma)
            alphaStar, _, _ = sp.fmin_l_bfgs_b(func=radial.fOpt,
                                               x0=np.zeros(radial.getDTRExtend().shape[1]),
                                               approx_grad=False,
                                               maxfun=15000,
                                               factr=1.0,
                                               bounds=[(0, C) for i in LTR],
                                               )

            resultRadial["config" + str(count)]["dualLoss"] = -radial.fOpt(alphaStar)[0][0]

            sllr = radial.computeScore(
                sVal=None,
                pEmp=None,
                alphaStar=alphaStar,
                D2=DVAL)
            Pval = (sllr > 0) * 1
            resultRadial["config" + str(count)]["errorRate"] = (Pval != LVAL).sum() / float(LVAL.size)
            resultRadial["config" + str(count)]["minDCF"] = minDCF(sllr, LVAL, 0.5, 1, 1)
            resultRadial["config" + str(count)]["actDCF"] = actDCF(sllr, LVAL, 0.5, 1, 1)

            count += 1

    if printResult:
        print("RESULT RADIAL SVM")
        for key in resultRadial.keys():
            print(
                f"\t{key} where K={resultRadial[key]["K"]} and C = {resultRadial[key]["C"]}, RBF(\u03B3 = {resultRadial[key]["gamma"]})")
            print(f"\t\tDual Loss: {resultRadial[key]["dualLoss"]:.6e}")
            print(f"\t\tError Rate: {resultRadial[key]["errorRate"] * 100:.1f}%")
            print(f"\t\tminDCF: {resultRadial[key]["minDCF"]:.4f}")
            print(f"\t\tactDCF: {resultRadial[key]["actDCF"]:.4f}")
            print()
