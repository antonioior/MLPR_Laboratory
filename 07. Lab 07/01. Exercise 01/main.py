import Commedia as cm
from Iris import computeConfusionMatrixIris
from graph import createRocCurve, createBayesErrorPlots
from load import load
from printValue import printConfusionMatrix
from utils import split_db_2to1, compute_Pfn_Pfp_allThresholds_fast

if __name__ == "__main__":
    printData = True
    graphRocCurve = True
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # CONFUSION MATRIX
    confusionMatrix_MVG, confusionMatrix_TCG = computeConfusionMatrixIris(DTR, LTR, DTE, LTE)

    confusionMatrix_Commedia = cm.computeConfusionMatrixCommedia()

    # OPTIMAL DECISIONS
    llrInf_Par, LTE_Inf_Par = cm.LoadllrLTEInfPar()
    configuration = {
        "one": [0.5, 1, 1],
        "two": [0.8, 1, 1],
        "three": [0.5, 10, 1],
        "four": [0.8, 1, 10],
    }
    confusionMatrixConfiguration = cm.computeConfusionMatrixForConfiguration(llrInf_Par, LTE_Inf_Par, configuration)

    # EVALUATION
    dcf, dcfNormalized = cm.computeDCFAndDCFNormalized(confusionMatrixConfiguration, configuration)

    # minimum detection costs
    minDCF = cm.comuputeMinDCF(llrInf_Par, LTE_Inf_Par, configuration)

    # ROC CURVES
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(llrInf_Par, LTE_Inf_Par)
    Ptp = 1 - Pfn

    # BAYES ERROR PLOTS
    effPriorLogOdds, dcfBayesError, minDCFBayesError = cm.bayesError(llrInf_Par, LTE_Inf_Par)

    if printData:
        print("MAIN - RESULT")
        # CONFUSION MATRIX
        print("\tCONFUSION MATRIX")
        print("\tConfusion matrix for MVG:")
        printConfusionMatrix(confusionMatrix_MVG)
        print("\tConfusion matrix for TCG:")
        printConfusionMatrix(confusionMatrix_TCG)
        print("\tConfusion matrix for Commedia:")
        printConfusionMatrix(confusionMatrix_Commedia)

        # OPTIMAL DECISIONS
        print("\tOPTIMAL DECISIONS")
        for key in confusionMatrixConfiguration:
            print(f"\tConfusion matrix for Inf_Par with configuration:\n"
                  f"\tpi = {configuration[key][0]}, Cfn = {configuration[key][1]}, Cfp = {configuration[key][2]}")
            printConfusionMatrix(confusionMatrixConfiguration[key])

        # EVALUATION
        print("\tEVALUATION")
        print("\t\t(pi, Cfn, Cfp) -> DCF")
        for key in dcf:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {dcf[key]: 3.3f}")

        print("\n\t\t(pi, Cfn, Cfp) -> DCFn")
        for key in dcfNormalized:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {dcfNormalized[key]: 3.3f}")

        print("\n\t\t(pi, Cfn, Cfp) -> min DCF")
        for key in minDCF:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {minDCF[key]: 3.3f}")

        # ROC CURVES
        createRocCurve(Pfp, Ptp, [0.0, 1.0], [0.0, 1.0], "FPR", "TPR")

        # BAYES ERROR PLOTS
        createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-3, 3], [0, 1.1])
