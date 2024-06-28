from MVG import MVG
from TCG import TCG
from utils import computeConfusionMatrix


def computeConfusionMatrixIris(DTR, LTR, DTE, LTE):
    # CONFUSION MATRIX FOR MVG
    PVAL_MVG = MVG(DTR, LTR, DTE, LTE, printData=False)
    confusionMatrix_MVG = computeConfusionMatrix(PVAL_MVG, LTE)

    # CONFUSION MATRIX FOR TCG
    PVAL_TCG = TCG(DTR, LTR, DTE, LTE, printData=False)
    confusionMatrix_TCG = computeConfusionMatrix(PVAL_TCG, LTE)
    return confusionMatrix_MVG, confusionMatrix_TCG
