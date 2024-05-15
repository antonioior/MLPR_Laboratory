import numpy as np

from printValue import printMatrix
from utils import vcol, logpdf_GAU_ND, compute_logPosterior


class meanAndCovarience:
    def __init__(self, mu, covariance):
        self.mu = mu
        self.covariance = covariance


def computeMeanAndCovariance(data):
    mu = data.mean(axis=1)
    mu = vcol(mu)
    dataCenter = data - mu
    covariance = (dataCenter @ dataCenter.T) / float(dataCenter.shape[1])
    covariance = covariance * np.eye(data.shape[0])
    return mu, covariance


def NB(DTR, LTR, DTE, LTE, printData=False):
    num_classes = np.unique(LTR).size
    mu_covariance = []
    S = np.zeros((num_classes, DTE.shape[1]))
    logPost_NB_pre_computed = np.load("../Solutions/logPosterior_NaiveBayes.npy")

    for i in range(num_classes):
        mu, covariance = computeMeanAndCovariance(DTR[:, LTR == i])
        mu_covariance.append(meanAndCovarience(mu, covariance))
        S[i, :] = logpdf_GAU_ND(DTE, mu, covariance)

    S_logPost = compute_logPosterior(S, np.ones(3) / 3.)
    maxAbsoluteError = np.abs(S_logPost - logPost_NB_pre_computed).max()
    PVAL = S_logPost.argmax(0)

    if printData:
        print("MAIN - RESULT OF NAIVE BAYES")
        for i in range(len(mu_covariance)):
            print(f"\tMean and covariance of property {i}:")
            print(f"\tMean:")
            printMatrix(mu_covariance[i].mu)
            print(f"\tCovarience:")
            printMatrix(mu_covariance[i].covariance)
        print(f"\tS_logPost_NB professor:")
        printMatrix(logPost_NB_pre_computed)
        print(f"\tS_logPost_NB:")
        printMatrix(S_logPost)
        print(f"\tMax absolute error w.r.t. pre-computed solution - log-posterior matrix")
        print(f"\t{maxAbsoluteError}")
        print(f"\tNB - Error rate: {(PVAL != LTE).sum() / float(LTE.size) * 100:.1f}")
