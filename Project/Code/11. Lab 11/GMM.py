import matplotlib.pyplot as plt

from DCF import minDCF, actDCF, bayesError
from LBG import LBGAlgorithm
from graph import createBayesErrorPlots
from utils import logpdf_GMM


def GMM(DTR, LTR, DVAL, LVAL, printResults=False):
    covTypes = ["full", "diagonal"]
    components = [1, 2, 4, 8, 16, 32]
    psi = 0.01
    alpha = 0.1
    priorT = 0.1
    resultGMM = []
    count = 0
    for covType in covTypes:
        resultGMM.append({
            "covType": covType,
            "values": []
        })
        for component in components:
            gmm0 = LBGAlgorithm(DTR[:, LTR == 0], alpha, component, psi=psi, covType=covType)
            gmm1 = LBGAlgorithm(DTR[:, LTR == 1], alpha, component, psi=psi, covType=covType)

            SLLR = logpdf_GMM(DVAL, gmm1)[1] - logpdf_GMM(DVAL, gmm0)[1]
            minDCFValue = minDCF(SLLR, LVAL, priorT, 1.0, 1.0)
            actDCFValue = actDCF(SLLR, LVAL, priorT, 1.0, 1.0)
            resultGMM[count]["values"].append({
                "component": component,
                "minDCF": minDCFValue,
                "actDCF": actDCFValue,
                "llr": SLLR
            })
        count += 1

    if printResults:
        lineLeft = -4
        lineRight = 4

        print("GMM RESULTS")
        for result in resultGMM:
            print(f"\t{result['covType'].upper()}")
            plt.figure(f"{result['covType'].upper()}", figsize=(15, 8), dpi=300)
            plt.suptitle(f"covType {result['covType'].upper()}")
            index = 1
            for value in result["values"]:
                print(
                    f"\t\tcomponent = {value['component']}, minDCF = {value['minDCF']:.4f}, actDCF = {value['actDCF']:.4f}")
                effPriorLogOdds, dcfBayesError, minDCFBayesError = bayesError(
                    llr=value["llr"],
                    LTE=LVAL,
                    lineLeft=lineLeft,
                    lineRight=lineRight
                )

                plt.subplot(2, 3, index)
                createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-4, 4], [0, 0.9], "r", "b",
                                      f"numComponent = {value["component"]}",
                                      show=False)
                index += 1
                plt.gca().set_xlim([lineLeft, lineRight])
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.tight_layout()
            plt.show()
