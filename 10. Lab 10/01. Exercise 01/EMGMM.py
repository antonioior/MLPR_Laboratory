import numpy as np

from utils import logpdf_GMM, vcol, vrow


def EMGmm(X, gmm, threshold=1e-6, maxIter=1000000):
    log_likelihood = []
    iteration = 0
    newGMM = []
    for _ in range(maxIter):
        responsability, logLikelihoodReturned = E_Step(X, gmm)
        gmm = M_Step(X, responsability, gmm)
        log_likelihood.append(logLikelihoodReturned)
        iteration += 1
        if len(log_likelihood) > 1 and abs(log_likelihood[-1] - log_likelihood[-2]) < threshold:
            newGMM = gmm
            break
    return log_likelihood, newGMM


def E_Step(X, gmm):
    S, logdens = logpdf_GMM(X, gmm)
    logResponsability = S - logdens
    responsability = np.exp(logResponsability)
    logLikelihood = logdens.mean()
    return responsability, logLikelihood


def M_Step(X, responsability, gmm):
    newGMM = []
    for gIndex in range(len(gmm)):
        gamma = responsability[gIndex]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        S = (vrow(gamma) * X) @ X.T
        mu = F / Z
        C = S / Z - mu @ mu.T
        w = Z / X.shape[1]
        newGMM.append((w, mu, C))
    return newGMM
