import numpy as np

from utils import computeConfusionMatrix, optimalBayesDecision, computeDCF, computeDCFNormalized, \
    compute_minDCF_binary_fast, computeEffectivePrior


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


def computeConfusionMatrixForConfiguration(llrInf_Par, LTE_Inf_Par, configuration):
    confusionMatrix = {
        "one": [[]],
        "two": [[]],
        "three": [[]],
        "four": [[]],

    }
    for key in configuration:
        PVAL_Inf_Par = computePVAL_Inf_Par(llrInf_Par, configuration[key][0], configuration[key][1],
                                           configuration[key][2])
        confusionMatrix[key] = computeConfusionMatrix(PVAL_Inf_Par, LTE_Inf_Par)
    return confusionMatrix


def computeDCFAndDCFNormalized(confusionMatrix, configuration):
    dcf, dcfNormalized = {}, {}
    for key in configuration:
        dcf[key] = computeDCF(
            confusionMatrix[key],
            configuration[key][0],
            configuration[key][1],
            configuration[key][2])
        dcfNormalized[key] = computeDCFNormalized(
            confusionMatrix[key],
            configuration[key][0],
            configuration[key][1],
            configuration[key][2])
    return dcf, dcfNormalized


def comuputeMinDCF(llrInf_Par, LTE_Inf_Par, configuration):
    minDCF = {}
    for key in configuration:
        minDCF[key] = compute_minDCF_binary_fast(
            llrInf_Par, LTE_Inf_Par, configuration[key][0],
            configuration[key][1],
            configuration[key][2])
    return minDCF


def bayesError(llrInf_Par, LTE_Inf_Par):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    piT = []
    matrix = []
    dcf = []
    minDCF = []
    for prior in effPriorLogOdds:
        piT.append(computeEffectivePrior(prior))

    for effectivePrior in piT:
        PVAL_Inf_Par = computePVAL_Inf_Par(llrInf_Par, effectivePrior, 1, 1)
        matrix = computeConfusionMatrix(PVAL_Inf_Par, LTE_Inf_Par)
        dcf.append(computeDCFNormalized(matrix, effectivePrior, 1, 1))
        minDCF.append(compute_minDCF_binary_fast(llrInf_Par, LTE_Inf_Par, effectivePrior, 1, 1))
    return effPriorLogOdds, dcf, minDCF
