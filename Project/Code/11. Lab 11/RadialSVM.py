import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF
from SVMClass import SVM
from graph import plotDCF, bayesError, createBayesErrorPlots
from graph import createMinDCFActDCFPlot


def radialSVM(DTR, LTR, DVAL, LVAL, printResult=True, titleGraph=""):
    resultRadial = []

    CValues = np.logspace(-3, 2, 11)
    K = np.sqrt(1)
    gammaValues = [1e-4, 1e-3, 1e-2, 1e-1]
    colorValue = ["blue", "green", "red", "cyan"]
    priorT = 0.1
    for gamma in range(len(gammaValues)):
        resultRadial.append((gamma, {}))
        count = 0
        for C in CValues:
            resultRadial[gamma][1]["config" + str(count)] = {
                "K": K,
                "C": C,
                "gamma": gammaValues[gamma]
            }
            radial = SVM(DTR, LTR, C, K, "radial", gamma=gammaValues[gamma])
            alphaStar, _, _ = sp.fmin_l_bfgs_b(func=radial.fOpt,
                                               x0=np.zeros(radial.getDTRExtend().shape[1]),
                                               approx_grad=False,
                                               maxfun=15000,
                                               factr=1.0,
                                               bounds=[(0, C) for i in LTR],
                                               )
            resultRadial[gamma][1]["config" + str(count)]["alphaStar"] = alphaStar
            resultRadial[gamma][1]["config" + str(count)]["dualLoss"] = -radial.fOpt(alphaStar)[0][0]

            sllr = radial.computeScore(
                alphaStar=alphaStar,
                D2=DVAL)
            Pval = (sllr > 0) * 1
            resultRadial[gamma][1]["config" + str(count)]["errorRate"] = (Pval != LVAL).sum() / float(LVAL.size)
            resultRadial[gamma][1]["config" + str(count)]["minDCF"] = minDCF(sllr, LVAL, priorT, 1, 1)
            resultRadial[gamma][1]["config" + str(count)]["actDCF"] = actDCF(sllr, LVAL, priorT, 1, 1)
            resultRadial[gamma][1]["config" + str(count)]["sllr"] = sllr
            count += 1

    if printResult:
        print()
        x = CValues
        yMinDCF = []
        yActDCF = []
        print("RESULT RADIAL SVM")
        for i in range(len(resultRadial)):
            print(f"\tFor \u03B3 index in vector {gammaValues[i]}")
            yMinDCF.append([])
            yActDCF.append([])
            for key in resultRadial[i][1]:
                print(
                    f"\t\t{key} where K={resultRadial[i][1][key]['K']} and C = {resultRadial[i][1][key]['C']}, RBF(\u03B3 =  {resultRadial[i][1][key]['gamma']})")
                print(f"\t\t\tDual Loss: {resultRadial[i][1][key]['dualLoss']:.6e}")
                print(f"\t\t\tError Rate: {resultRadial[i][1][key]['errorRate'] * 100:.1f}%")
                print(f"\t\t\tminDCF: {resultRadial[i][1][key]['minDCF']:.4f}")
                print(f"\t\t\tactDCF: {resultRadial[i][1][key]['actDCF']:.4f}")
                print()
                yMinDCF[i].append(resultRadial[i][1][key]["minDCF"])
                yActDCF[i].append(resultRadial[i][1][key]["actDCF"])

        for i in range(len(resultRadial)):
            plotDCF(
                x=x,
                y=yMinDCF[i],
                xLabel="C",
                yLabel="minDCF value",
                label=f"gamma = {gammaValues[i]}",
                color=colorValue[i],
                title=titleGraph,
                logScale=True,
                show=False
            )
        plt.show()

        for i in range(len(resultRadial)):
            plotDCF(
                x=x,
                y=yActDCF[i],
                xLabel="C",
                yLabel="actDCF value",
                label=f"gamma = {gammaValues[i]}",
                color=colorValue[i],
                title=titleGraph,
                logScale=True,
                show=False
            )
        plt.show()

        createMinDCFActDCFPlot(
            x=x,
            yMinDCF=yMinDCF[3],  # index 3 is the last gamma value gamma = 0.1
            yActDCF=yActDCF[3],
            xLabel="C",
            colorMinDCF="blue",
            colorActDCF="red",
            title=titleGraph,
            togheter=True,
            logScale=True
        )

        createMinDCFActDCFPlot(
            x=x,
            yMinDCF=yMinDCF[3],
            yActDCF=yActDCF[3],
            xLabel="C",
            colorMinDCF="blue",
            colorActDCF="red",
            title=titleGraph,
            togheter=False,
            logScale=True
        )

        for key in resultRadial[3][1]:
            if resultRadial[3][1][key]['C'] == 100 and resultRadial[3][1][key]['gamma'] == gammaValues[3]:
                effPriorLogOdds, dcfBayesError, minDCFBayesError = bayesError(
                    llr=resultRadial[3][1][key]['sllr'],
                    LTE=LVAL,
                    lineLeft=-4,
                    lineRight=4
                )
                createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-4, 4], [0, 0.9], "r", "b",
                                      f"SVM with RBF kernel, C = {resultRadial[3][1][key]['C']}, \u03B3 = {resultRadial[3][1][key]['gamma']}",
                                      show=True)
