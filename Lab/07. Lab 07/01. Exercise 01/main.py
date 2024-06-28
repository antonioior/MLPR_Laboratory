import matplotlib.pyplot as plt
import numpy as np

import Commedia as cm
from Iris import computeConfusionMatrixIris
from graph import createRocCurve, createBayesErrorPlots
from load import load
from printValue import printConfusionMatrix
from utils import split_db_2to1, compute_Pfn_Pfp_allThresholds_fast

if __name__ == "__main__":
    printData = True
    printGraph = True
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # CONFUSION MATRIX
    confusionMatrix_MVG, confusionMatrix_TCG = computeConfusionMatrixIris(DTR, LTR, DTE, LTE)

    confusionMatrix_Commedia = cm.computeConfusionMatrixCommedia()

    # OPTIMAL DECISIONS
    llrInf_Par, LTE_Inf_Par, llrInf_Par_Eps1 = cm.LoadllrLTEInfPar()
    configuration = {
        "one": [0.5, 1, 1],
        "two": [0.8, 1, 1],
        "three": [0.5, 10, 1],
        "four": [0.8, 1, 10],
    }
    confusionMatrixConfiguration = cm.computeConfusionMatrixForConfiguration(llrInf_Par, LTE_Inf_Par, configuration)
    confusionMatrixConfigurationEps1 = cm.computeConfusionMatrixForConfiguration(llrInf_Par_Eps1, LTE_Inf_Par,
                                                                                 configuration)

    # EVALUATION
    dcf, dcfNormalized = cm.computeDCFAndDCFNormalized(confusionMatrixConfiguration, configuration)
    dcfEps1, dcfNormalizedEps1 = cm.computeDCFAndDCFNormalized(confusionMatrixConfigurationEps1, configuration)

    # minimum detection costs
    minDCF = cm.comuputeMinDCF(llrInf_Par, LTE_Inf_Par, configuration)
    minDCFEps1 = cm.comuputeMinDCF(llrInf_Par_Eps1, LTE_Inf_Par, configuration)

    # ROC CURVES
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(llrInf_Par, LTE_Inf_Par)
    Ptp = 1 - Pfn

    # BAYES ERROR PLOTS
    effPriorLogOdds, dcfBayesError, minDCFBayesError = cm.bayesError(llrInf_Par, LTE_Inf_Par)

    effPriorLogOddsEps1, dcfBayesErrorEps1, minDCFBayesErrorEps1 = cm.bayesError(llrInf_Par_Eps1, LTE_Inf_Par)

    # MULTICLASS EVALUATION
    C = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    prior = np.array([0.3, 0.4, 0.3])

    # epsilon = 0.001
    S_logPost, LTE_Commedia = cm.loadMulticlass()
    confusionMatrixMulticlass = cm.computeConfusionMatrixMulticlass(S_logPost, prior, C, LTE_Commedia)
    DCFMulticlass, DCFMulticlassNormalized = cm.computeDCFMuilticlassCommedia(confusionMatrixMulticlass, prior, C)

    # epsilon = 1
    S_logPostEps1, LTE_CommediaEps1 = cm.loadMulticlassEps1()
    confusionMatrixMulticlassEps1 = cm.computeConfusionMatrixMulticlass(S_logPostEps1, prior, C, LTE_CommediaEps1)
    DCFMulticlassEps1, DCFMulticlassNormalizedEps1 = cm.computeDCFMuilticlassCommedia(confusionMatrixMulticlassEps1,
                                                                                      prior,
                                                                                      C)

    # uniformClass
    CUniform = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    priorUniform = np.array([1 / 3, 1 / 3, 1 / 3])

    confusionMatrixMulticlassUniform = cm.computeConfusionMatrixMulticlass(S_logPost, priorUniform, CUniform,
                                                                           LTE_Commedia)
    DCFMulticlassUniform, DCFMulticlassNormalizedUniform = cm.computeDCFMuilticlassCommedia(
        confusionMatrixMulticlassUniform, priorUniform, CUniform)

    confusionMatrixMulticlassEps1Uniform = cm.computeConfusionMatrixMulticlass(S_logPostEps1, priorUniform, CUniform,
                                                                               LTE_CommediaEps1)
    DCFMulticlassEps1Uniform, DCFMulticlassNormalizedEps1Uniform = cm.computeDCFMuilticlassCommedia(
        confusionMatrixMulticlassEps1Uniform,
        priorUniform,
        CUniform)

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

        print("\n\t\t(pi, Cfn, Cfp) -> DCFn \u03B5 = 0.001")
        for key in dcfNormalized:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {dcfNormalized[key]: 3.3f}")

        print("\n\t\t(pi, Cfn, Cfp) -> min DCF \u03B5 = 0.001")
        for key in minDCF:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {minDCF[key]: 3.3f}")

        if printGraph:
            # ROC CURVES
            createRocCurve(Pfp, Ptp, [0.0, 1.0], [0.0, 1.0], "FPR", "TPR")

            # BAYES ERROR PLOTS EPSILON = 0.001
            createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-3, 3], [0, 1.1], 0.001, "r", "b")
            plt.show()

            # BAYES ERROR PLOTS EPSILON = 1
            createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-3, 3], [0, 1.1], 0.001, "r", "b")
            createBayesErrorPlots(effPriorLogOddsEps1, dcfBayesErrorEps1, minDCFBayesErrorEps1, [-3, 3], [0, 1.1], 1,
                                  "orange",
                                  "cyan")
            plt.show()

        print("\n\t\t(pi, Cfn, Cfp) -> DCFn \u03B5 = 1")
        for key in dcfNormalizedEps1:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {dcfNormalizedEps1[key]: 3.3f}")

        print("\n\t\t(pi, Cfn, Cfp) -> min DCF \u03B5 =1")
        for key in minDCFEps1:
            print(
                f"\t\t{configuration[key][0], configuration[key][1], configuration[key][2]} -> {minDCFEps1[key]: 3.3f}")

        print("\n\tMULTICLASS EVALUATION for \u03B5 = 0.001")
        printConfusionMatrix(confusionMatrixMulticlass)
        print(f"\t\tDCF = {DCFMulticlass:.3f}")
        print(f"\t\tDCF Normalized = {DCFMulticlassNormalized:.3f}")
        print("\n\tMULTICLASS EVALUATION for \u03B5 = 1")
        printConfusionMatrix(confusionMatrixMulticlassEps1)
        print(f"\t\tDCF = {DCFMulticlassEps1:.3f}")
        print(f"\t\tDCF Normalized = {DCFMulticlassNormalizedEps1:.3f}")

        print("\n\tMULTICLASS EVALUATION for Uniform")
        print("\t\t\t\t\t DCF\tDCFNormalized")
        print(f"\t \u03B5 = 0.001\t\t{DCFMulticlassUniform:.3f}\t  {DCFMulticlassNormalizedUniform:.3f}")
        print(f"\t \u03B5 = 1\t\t\t{DCFMulticlassEps1Uniform:.3f}\t  {DCFMulticlassNormalizedEps1Uniform:.3f}")
