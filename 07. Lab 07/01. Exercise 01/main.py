from Commedia import computeConfusionMatrixCommedia, LoadllrLTEInfPar, computePVAL_Inf_Par
from Iris import computeConfusionMatrixIris
from graph import createRocCurve
from load import load
from printValue import printConfusionMatrix
from utils import split_db_2to1, computeConfusionMatrix, computeDCF, computeDCFNormalized, compute_minDCF_binary_fast, \
    compute_Pfn_Pfp_allThresholds_fast

if __name__ == "__main__":
    printData = True
    graphRocCurve = True
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # CONFUSION MATRIX
    confusionMatrix_MVG, confusionMatrix_TCG = computeConfusionMatrixIris(DTR, LTR, DTE, LTE)

    confusionMatrix_Commedia = computeConfusionMatrixCommedia()

    # OPTIMAL DECISIONS
    llrInf_Par, LTE_Inf_Par = LoadllrLTEInfPar()
    valueConfiguration = {
        "one": [0.5, 1, 1],
        "two": [0.8, 1, 1],
        "three": [0.5, 10, 1],
        "four": [0.8, 1, 10],
    }
    valueConfusionMatrix = {
        "one": [[]],
        "two": [[]],
        "three": [[]],
        "four": [[]],

    }
    for key in valueConfiguration:
        PVAL_Inf_Par = computePVAL_Inf_Par(llrInf_Par, valueConfiguration[key][0], valueConfiguration[key][1],
                                           valueConfiguration[key][2])
        valueConfusionMatrix[key] = computeConfusionMatrix(PVAL_Inf_Par, LTE_Inf_Par)

    # EVALUATION
    dcf = {}
    dcfNormalized = {}
    for key in valueConfiguration:
        dcf[key] = computeDCF(
            valueConfusionMatrix[key],
            valueConfiguration[key][0],
            valueConfiguration[key][1],
            valueConfiguration[key][2])
        dcfNormalized[key] = computeDCFNormalized(
            valueConfusionMatrix[key],
            valueConfiguration[key][0],
            valueConfiguration[key][1],
            valueConfiguration[key][2])

    # minimum detection costs
    minDCF = {}
    for key in valueConfiguration:
        minDCF[key] = compute_minDCF_binary_fast(llrInf_Par, LTE_Inf_Par, valueConfiguration[key][0],
                                                 valueConfiguration[key][1],
                                                 valueConfiguration[key][2])

    # ROC CURVES
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(llrInf_Par, LTE_Inf_Par)
    Ptp = 1 - Pfn
    if graphRocCurve:
        createRocCurve(Pfp, Ptp, [0.0, 1.0], [0.0, 1.0], "FPR", "TPR")

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
        for key in valueConfusionMatrix:
            print(f"\tConfusion matrix for Inf_Par with configuration:\n"
                  f"\tpi = {valueConfiguration[key][0]}, Cfn = {valueConfiguration[key][1]}, Cfp = {valueConfiguration[key][2]}")
            printConfusionMatrix(valueConfusionMatrix[key])

        # EVALUATION
        print("\tEVALUATION")
        print("\t\t(pi, Cfn, Cfp) -> DCF")
        for key in dcf:
            print(
                f"\t\t{valueConfiguration[key][0], valueConfiguration[key][1], valueConfiguration[key][2]} -> {dcf[key]: 3.3f}")

        print("\n\t\t(pi, Cfn, Cfp) -> DCFn")
        for key in dcfNormalized:
            print(
                f"\t\t{valueConfiguration[key][0], valueConfiguration[key][1], valueConfiguration[key][2]} -> {dcfNormalized[key]: 3.3f}")

        print("\n\t\t(pi, Cfn, Cfp) -> min DCF")
        for key in minDCF:
            print(
                f"\t\t{valueConfiguration[key][0], valueConfiguration[key][1], valueConfiguration[key][2]} -> {minDCF[key]: 3.3f}")
