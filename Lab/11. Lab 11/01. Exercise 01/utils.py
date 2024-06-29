import numpy as np


# LAB 07
def optimalBayesDecision(pi, Cfn, Cfp):
    threshold = - np.log(pi * Cfn / ((1 - pi) * Cfp))
    return threshold


def computePVAL(llr, pi, Cfn, Cfp):
    t = optimalBayesDecision(pi, Cfn, Cfp)
    PVAL = np.zeros(len(llr), dtype=int)
    for i in range(len(llr)):
        PVAL[i] = 1 if llr[i] > t else 0
    return PVAL


def computeConfusionMatrix(PVAL, LTE):
    num_classes = len(set(LTE))
    confusionMatrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(PVAL)):
        confusionMatrix[PVAL[i]][LTE[i]] += 1
    return confusionMatrix
