import numpy as np

from utils import computeConfusionMatrix, optimalBayesDecision


def computeConfusionMatrixCommedia():
    S_logPost = np.load("../Data/commedia_ll.npy")
    PVAL_Commedia = S_logPost.argmax(0)
    LTE_Commedia = np.load("../Data/commedia_labels.npy")
    confusionMatrix = computeConfusionMatrix(PVAL_Commedia, LTE_Commedia)
    return confusionMatrix


def LoadllrLTEInfPar():
    llrInf_Par = np.load("../Data/commedia_llr_infpar.npy")
    LTE_Inf_Par = np.load("../Data/commedia_labels_infpar.npy")
    return llrInf_Par, LTE_Inf_Par


def computePVAL_Inf_Par(llrInf_Par, pi, Cfn, Cfp):
    t = optimalBayesDecision(pi, Cfn, Cfp)
    PVAL_Inf_Par = np.zeros(len(llrInf_Par), dtype=int)
    for i in range(len(llrInf_Par)):
        PVAL_Inf_Par[i] = 1 if llrInf_Par[i] > t else 0
    return PVAL_Inf_Par
