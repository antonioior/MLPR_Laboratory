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
    return mu, covariance


def MVG(DTR, LTR, DTE, LTE, printData=False):
    num_classes = np.unique(LTR).size
    mu_covariance = []
    S = np.zeros((num_classes, DTE.shape[1]))

    for i in range(num_classes):
        mu, covariance = computeMeanAndCovariance(DTR[:, LTR == i])
        mu_covariance.append(meanAndCovarience(mu, covariance))
        S[i, :] = logpdf_GAU_ND(DTE, mu, covariance)

    S_logPost = compute_logPosterior(S, np.ones(3) / 3.)
    PVAL = S_logPost.argmax(0)

    if printData:
        print("MAIN - RESULT OF MVG")
        for i in range(len(mu_covariance)):
            print(f"\tMean and covariance of property {i}:")
            print(f"\tMean:")
            printMatrix(mu_covariance[i].mu)
            print(f"\tCovarience:")
            printMatrix(mu_covariance[i].covariance)
        print(f"\tS_logPost_MVG:")
        printMatrix(S_logPost)
        print(f"\tError rate: {(PVAL != LTE).sum() / float(LTE.size) * 100:.1f}")

    return PVAL
