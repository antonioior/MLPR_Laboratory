import numpy as np

from printValue import printMatrix
from utils import vcol, logpdf_GAU_ND, compute_logPosterior


class Mean():
    def __init__(self, mu):
        self.mu = mu


def computeMeanAndCovariance(data):
    mu = data.mean(axis=1)
    mu = vcol(mu)
    dataCenter = data - mu
    covariance = (dataCenter @ dataCenter.T) / float(dataCenter.shape[1])
    covariance = covariance * np.eye(data.shape[0])
    return mu, covariance


def computeMean(data):
    mu = data.mean(axis=1)
    mu = vcol(mu)
    return mu


def TCG(DTR, LTR, DTE, LTE, printData=False):
    num_classes = np.unique(LTR).size
    product = []
    means = []
    S = np.zeros((num_classes, DTE.shape[1]))
    logPost_TCG_pre_computed = np.load("../Solutions/logPosterior_TiedMVG.npy")

    for i in range(num_classes):
        mu = computeMean(DTR[:, LTR == i])
        means.append(Mean(mu))
        dataCenter = DTR[:, LTR == i] - mu
        product.append(dataCenter @ dataCenter.T)
    covariance = sum(product) / float(DTR.shape[1])

    for i in range(num_classes):
        S[i, :] = logpdf_GAU_ND(DTE, means[i].mu, covariance)

    S_logPost = compute_logPosterior(S, np.ones(3) / 3.)
    maxAbsoluteError = np.abs(S_logPost - logPost_TCG_pre_computed).max()
    PVAL = S_logPost.argmax(0)

    if printData:
        print("MAIN - RESULT OF TIED COVARIANCE GAUSSIAN CLASSIFIER")
        for i in range(len(means)):
            print(f"\tMean and covariance of property {i}:")
            print(f"\tMean:")
            printMatrix(means[i].mu)
        print(f"\tCovarience:")
        printMatrix(covariance)
        print(f"\tS_logPost_TCG professor:")
        printMatrix(logPost_TCG_pre_computed)
        print(f"\tS_logPost_TCG:")
        printMatrix(S_logPost)
        print(f"\tMax absolute error w.r.t. pre-computed solution - log-posterior matrix")
        print(f"\t{maxAbsoluteError}")
        print(f"\tError rate: {(PVAL != LTE).sum() / float(LTE.size) * 100:.1f}%")
