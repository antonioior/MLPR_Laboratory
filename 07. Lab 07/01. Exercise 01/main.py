from Commedia import computeConfusionMatrixCommedia, LoadllrLTEInfPar, computePVAL_Inf_Par
from Iris import computeConfusionMatrixIris
from load import load
from printValue import printConfusionMatrix
from utils import split_db_2to1, computeConfusionMatrix

if __name__ == "__main__":
    printData = True
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # CONFUSION MATRIX
    confusionMatrix_MVG, confusionMatrix_TCG = computeConfusionMatrixIris(DTR, LTR, DTE, LTE)

    confusionMatrix_Commedia = computeConfusionMatrixCommedia()

    # Optimal Decision
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

    if printData:
        print("MAIN - RESULT")
        print("\tCONFUSION MATRIX")
        print("\tConfusion matrix for MVG:")
        printConfusionMatrix(confusionMatrix_MVG)
        print("\tConfusion matrix for TCG:")
        printConfusionMatrix(confusionMatrix_TCG)
        print("\tConfusion matrix for Commedia:")
        printConfusionMatrix(confusionMatrix_Commedia)

        print("\tOPTIMAL DECISIONS")
        for key in valueConfusionMatrix:
            print(f"\tConfusion matrix for Inf_Par with configuration:\n"
                  f"\tpi = {valueConfiguration[key][0]}, Cfn = {valueConfiguration[key][1]}, Cfp = {valueConfiguration[key][2]}")
            printConfusionMatrix(valueConfusionMatrix[key])
