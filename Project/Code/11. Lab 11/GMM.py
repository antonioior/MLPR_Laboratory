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
        for componentGmm0 in components:
            for componentGmm1 in components:
                gmm0 = LBGAlgorithm(DTR[:, LTR == 0], alpha, componentGmm0, psi=psi, covType=covType)
                gmm1 = LBGAlgorithm(DTR[:, LTR == 1], alpha, componentGmm1, psi=psi, covType=covType)

                SLLR = logpdf_GMM(DVAL, gmm1)[1] - logpdf_GMM(DVAL, gmm0)[1]
                minDCFValue = minDCF(SLLR, LVAL, priorT, 1.0, 1.0)
                actDCFValue = actDCF(SLLR, LVAL, priorT, 1.0, 1.0)
                resultGMM[count]["values"].append({
                    "componentGmm0": componentGmm0,
                    "componentGmm1": componentGmm1,
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
            plt.figure(f"{result['covType'].upper()}", figsize=(30, 16), dpi=300)
            plt.suptitle(f"covType {result['covType'].upper()}")
            index = 1
            for value in result["values"]:
                print(
                    f"\t\tcomponentGmm0 = {value['componentGmm0']}, componentGmm1 = {value['componentGmm1']}, minDCF = {value['minDCF']:.4f}, actDCF = {value['actDCF']:.4f}")
                effPriorLogOdds, dcfBayesError, minDCFBayesError = bayesError(
                    llr=value["llr"],
                    LTE=LVAL,
                    lineLeft=lineLeft,
                    lineRight=lineRight
                )

                plt.subplot(6, 6, index)
                createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-4, 4], [0, 0.9], "r", "b",
                                      f"componentGmm0 = {value["componentGmm0"]}, componentGmm1 = {value["componentGmm1"]}",
                                      show=False)
                index += 1
                plt.gca().set_xlim([lineLeft, lineRight])
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.tight_layout()
            plt.show()


# LAB 11
class GMMObject:
    def __init__(self, DTR, LTR, componentGMM0, componentGMM1, alpha, psi, covType):
        self.DTR = DTR
        self.LTR = LTR
        self.componentGMM0 = componentGMM0
        self.componentGMM1 = componentGMM1
        self.alpha = alpha
        self.psi = psi
        self.covType = covType

    def trainGMMReturnMinAndActDCF(self, DVAL, LVAL, priorT):
        self.gmm0 = LBGAlgorithm(self.DTR[:, self.LTR == 0], self.alpha, self.componentGMM0, psi=self.psi,
                                 covType=self.covType)
        self.gmm1 = LBGAlgorithm(self.DTR[:, self.LTR == 1], self.alpha, self.componentGMM1, psi=self.psi,
                                 covType=self.covType)
        sllr = logpdf_GMM(DVAL, self.gmm1)[1] - logpdf_GMM(DVAL, self.gmm0)[1]
        minDCFWithoutCal = minDCF(sllr, LVAL, priorT, 1.0, 1.0)
        actDCFWithoutCal = actDCF(sllr, LVAL, priorT, 1.0, 1.0)
        return sllr, minDCFWithoutCal, actDCFWithoutCal

    def computeScore(self, Dtest):
        return logpdf_GMM(Dtest, self.gmm1)[1] - logpdf_GMM(Dtest, self.gmm0)[1]
