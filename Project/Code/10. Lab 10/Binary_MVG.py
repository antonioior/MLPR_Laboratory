import numpy as np

from printValue import printMatrix
from utils import vcol, logpdf_GAU_ND_prof


class meanAndCovarience:
    def __init__(self, mu, covariance):
        self.mu = mu
        self.covariance = covariance


class Mean():
    def __init__(self, mu):
        self.mu = mu


def computeMean(data):
    mu = data.mean(axis=1)
    mu = vcol(mu)
    return mu


def computeMeanAndCovariance(data):
    mu = data.mean(axis=1)
    mu = vcol(mu)
    dataCenter = data - mu
    covariance = (dataCenter @ dataCenter.T) / float(dataCenter.shape[1])
    return mu, covariance


def Binary_MVG(DTR, LTR, DTE, LTE, printData=False):
    labelSet = set(LTR)
    mu_covariance = []
    S = np.zeros((len(labelSet), DTE.shape[1]))
    for i in labelSet:
        mu, covariance = computeMeanAndCovariance(DTR[:, LTR == i])
        mu_covariance.append(meanAndCovarience(mu, covariance))
        S[i, :] = logpdf_GAU_ND_prof(DTE, mu, covariance)

    llr = S[1, :] - S[0, :]
    PVAL = np.zeros(DTE.shape[1], dtype=np.int32)
    TH = 0
    PVAL[llr >= TH] = 1
    PVAL[llr < TH] = 0
    errorRate = (PVAL != LTE).sum() / float(LTE.size)
    if printData:
        print("MAIN - RESULT OF BINARY MVG")
        for i in range(len(mu_covariance)):
            print(f"\tMean and covariance of property {i}:")
            print(f"\tMean:")
            printMatrix(mu_covariance[i].mu)
            print(f"\tCovarience:")
            printMatrix(mu_covariance[i].covariance)

        print(f"\tLLR:\n\t{llr}")
        print(f"\tError rate: {errorRate * 100:.1f}")
    return errorRate, llr


def Binary_Tied(DTR, LTR, DTE, LTE, printData=False):
    labelSet = set(LTR)
    product = []
    means = []
    S = np.zeros((len(labelSet), DTE.shape[1]))

    for i in labelSet:
        mu = computeMean(DTR[:, LTR == i])
        means.append(Mean(mu))
        dataCenter = DTR[:, LTR == i] - mu
        product.append(dataCenter @ dataCenter.T)
    covariance = sum(product) / float(DTR.shape[1])

    for i in labelSet:
        S[i, :] = logpdf_GAU_ND_prof(DTE, means[i].mu, covariance)

    llr = S[1, :] - S[0, :]
    PVAL = np.zeros(DTE.shape[1], dtype=np.int32)
    TH = 0
    PVAL[llr >= TH] = 1
    PVAL[llr < TH] = 0
    errorRate = (PVAL != LTE).sum() / float(LTE.size)
    if printData:
        print("MAIN - RESULT OF BINARY TIED COVARIANCE GAUSSIAN CLASSIFIER")
        for i in range(len(means)):
            print(f"\tMean and covariance of property {i}:")
            print(f"\tMean:")
            printMatrix(means[i].mu)
        print(f"\tCovarience:")
        printMatrix(covariance)
        print(f"\tLLR:\n\t{llr}")
        print(f"\tError rate: {errorRate * 100:.1f}")

    return errorRate, llr
