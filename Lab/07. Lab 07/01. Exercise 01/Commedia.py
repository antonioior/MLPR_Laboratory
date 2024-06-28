import numpy as np

import utils as ut


def computeConfusionMatrixCommedia():
    S_logPost = np.load("../Data/commedia_ll.npy")
    PVAL_Commedia = S_logPost.argmax(0)
    LTE_Commedia = np.load("../Data/commedia_labels.npy")
    confusionMatrix = ut.computeConfusionMatrix(PVAL_Commedia, LTE_Commedia)
    return confusionMatrix


def LoadllrLTEInfPar():
    llrInf_Par = np.load("../Data/commedia_llr_infpar.npy")
    LTE_Inf_Par = np.load("../Data/commedia_labels_infpar.npy")
    llrInf_Par_Eps1 = np.load("../Data/commedia_llr_infpar_eps1.npy")
    return llrInf_Par, LTE_Inf_Par, llrInf_Par_Eps1


def computePVAL_Inf_Par(llrInf_Par, pi, Cfn, Cfp):
    t = ut.optimalBayesDecision(pi, Cfn, Cfp)
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
        confusionMatrix[key] = ut.computeConfusionMatrix(PVAL_Inf_Par, LTE_Inf_Par)
    return confusionMatrix


def computeDCFAndDCFNormalized(confusionMatrix, configuration):
    dcf, dcfNormalized = {}, {}
    for key in configuration:
        dcf[key] = ut.computeDCF(
            confusionMatrix[key],
            configuration[key][0],
            configuration[key][1],
            configuration[key][2])
        dcfNormalized[key] = ut.computeDCFNormalized(
            confusionMatrix[key],
            configuration[key][0],
            configuration[key][1],
            configuration[key][2])
    return dcf, dcfNormalized


def comuputeMinDCF(llrInf_Par, LTE_Inf_Par, configuration):
    minDCF = {}
    for key in configuration:
        minDCF[key] = ut.compute_minDCF_binary_fast(
            llrInf_Par, LTE_Inf_Par, configuration[key][0],
            configuration[key][1],
            configuration[key][2])
    return minDCF


def bayesError(llrInf_Par, LTE_Inf_Par):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    piT = []
    dcf = []
    minDCF = []
    for prior in effPriorLogOdds:
        piT.append(ut.computeEffectivePrior(prior))

    for effectivePrior in piT:
        PVAL_Inf_Par = computePVAL_Inf_Par(llrInf_Par, effectivePrior, 1, 1)
        matrix = ut.computeConfusionMatrix(PVAL_Inf_Par, LTE_Inf_Par)
        dcf.append(ut.computeDCFNormalized(matrix, effectivePrior, 1, 1))
        minDCF.append(ut.compute_minDCF_binary_fast(llrInf_Par, LTE_Inf_Par, effectivePrior, 1, 1))
    return effPriorLogOdds, dcf, minDCF


def loadMulticlass():
    S_logPost = np.load("../Data/commedia_ll.npy")
    LTE_Commedia = np.load("../Data/commedia_labels.npy")
    return S_logPost, LTE_Commedia


def computeConfusionMatrixMulticlass(S_logPost, prior, C, LTE_Commedia):
    commedia_posteriors = ut.compute_posteriors(S_logPost, prior)
    PVAL_Commedia = ut.compute_optimal_Bayes(commedia_posteriors, C)
    return ut.computeConfusionMatrix(PVAL_Commedia, LTE_Commedia)


def computeDCFMuilticlassCommedia(confusionMatrix, prior, C):
    DCFMulticlass = ut.compute_empirical_Bayes_risk(confusionMatrix, prior, C, normalize=False)
    DCFMutliclassN = ut.compute_empirical_Bayes_risk(confusionMatrix, prior, C, normalize=True)
    return DCFMulticlass, DCFMutliclassN


def loadMulticlassEps1():
    S_logPost = np.load("../Data/commedia_ll_eps1.npy")
    LTE_Commedia = np.load("../Data/commedia_labels.npy")
    return S_logPost, LTE_Commedia
