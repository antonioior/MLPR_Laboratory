import numpy as np
from utils import computePVAL, computeConfusionMatrix
#Lab 07
def computeDCF(confusionMatrix, pi, Cfn, Cfp, normalized=True):
    Pfn = confusionMatrix[0][1] / (confusionMatrix[1][1] + confusionMatrix[0][1])
    Pfp = confusionMatrix[1][0] / (confusionMatrix[0][0] + confusionMatrix[1][0])
    dcfValue = pi * Cfn * Pfn + (1 - pi) * Cfp * Pfp
    if normalized:
        Bdummy = min(pi * Cfn, (1 - pi) * Cfp)
        return dcfValue / Bdummy
    return dcfValue


def computeEffectivePrior(prior):
    return 1 / (1 + np.exp(-prior))

def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]  # We sort the llrs
    classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []

    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse

    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)

    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
            idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])

    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)  # we return also the corresponding thresholds


# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (
            1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


def bayesError(llr, LTE, lineLeft, lineRight):
    effPriorLogOdds = np.linspace(lineLeft, lineRight, 21)
    piT = []
    dcf = []
    minDCF = []
    for prior in effPriorLogOdds:
        piT.append(computeEffectivePrior(prior))

    for effectivePrior in piT:
        PVAL = computePVAL(llr, effectivePrior, 1, 1)
        matrix = computeConfusionMatrix(PVAL, LTE)
        dcf.append(computeDCF(matrix, effectivePrior, 1, 1, True))
        minDCF.append(compute_minDCF_binary_fast(llr, LTE, effectivePrior, 1, 1))
    return effPriorLogOdds, dcf, minDCF