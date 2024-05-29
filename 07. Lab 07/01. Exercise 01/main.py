from Commedia import computeConfusionMatrixCommedia
from Iris import computeConfusionMatrixIris
from load import load
from printValue import printConfusionMatrix
from utils import split_db_2to1

if __name__ == "__main__":
    printData = True
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    confusionMatrix_MVG, confusionMatrix_TCG = computeConfusionMatrixIris(DTR, LTR, DTE, LTE)

    confusionMatrix_Commedia = computeConfusionMatrixCommedia()

    if printData:
        print("MAIN - RESULT")
        print("\tConfusion matrix for MVG:")
        printConfusionMatrix(confusionMatrix_MVG)
        print("\tConfusion matrix for TCG:")
        printConfusionMatrix(confusionMatrix_TCG)
        print("\tConfusion matrix for Commedia:")
        printConfusionMatrix(confusionMatrix_Commedia)
