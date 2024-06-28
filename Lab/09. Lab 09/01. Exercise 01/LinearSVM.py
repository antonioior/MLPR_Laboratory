import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from SVMClass import SVM
from utils import errorRate


def linearSVM(DTR, LTR, DVAL, LVAL, printResult=False):
    resultLinear = {}
    count = 0
    for K in [1, 10]:
        for C in [0.1, 1, 10]:
            resultLinear["config" + str(count)] = {
                "K": K,
                "C": C
            }
            linear = SVM(DTR, LTR, C, K, "linear")
            alphaStar, _, _ = sp.fmin_l_bfgs_b(func=linear.fOpt,
                                               x0=np.zeros(linear.getDTRExtend().shape[1]),
                                               approx_grad=False,
                                               maxfun=15000,
                                               factr=1.0,
                                               bounds=[(0, C) for i in LTR],
                                               )
            WHat = linear.computeWHat(alphaStar)
            resultLinear["config" + str(count)]["primalLoss"] = linear.primalLoss(WHat)
            resultLinear["config" + str(count)]["dualLoss"] = -linear.fOpt(alphaStar)[0][0]
            resultLinear["config" + str(count)]["dualityGap"] = resultLinear["config" + str(count)]["primalLoss"] - \
                                                                resultLinear["config" + str(count)]["dualLoss"]
            resultLinear["config" + str(count)]["errorRate"], sVal = errorRate(DVAL, LVAL, WHat, K)
            resultLinear["config" + str(count)]["minDCF"] = minDCF(sVal, LVAL, 0.5, 1, 1)
            resultLinear["config" + str(count)]["actDCF"] = actDCF(sVal, LVAL, 0.5, 1, 1)
            count += 1

    if printResult:
        print("RESULT LINEAR SVM")
        for key in resultLinear.keys():
            print(f"\t{key} where k={resultLinear[key]["K"]} and C = {resultLinear[key]["C"]}")
            print(f"\t\tPrimal Loss: {resultLinear[key]["primalLoss"]:.6e}")
            print(f"\t\tDual Loss: {resultLinear[key]["dualLoss"]:.6e}")
            print(f"\t\tDuality Gap: {resultLinear[key]["dualityGap"]:.6e}")
            print(f"\t\tError Rate: {resultLinear[key]["errorRate"] * 100 :.1f}%")
            print(f"\t\tminDCF: {resultLinear[key]["minDCF"]:.4f}")
            print(f"\t\tactDCF: {resultLinear[key]["actDCF"]:.4f}")
            print()
