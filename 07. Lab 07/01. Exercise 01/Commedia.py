import numpy as np

from utils import computeConfusionMatrix


def computeConfusionMatrixCommedia():
    S_logPost = np.load("../Data/commedia_ll.npy")
    PVAL_Commedia = S_logPost.argmax(0)
    LTE_Commedia = np.load("../Data/commedia_labels.npy")
    confusionMatrix = computeConfusionMatrix(PVAL_Commedia, LTE_Commedia)
    return confusionMatrix
