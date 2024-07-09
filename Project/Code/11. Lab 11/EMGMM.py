import numpy as np

from utils import logpdf_GMM, vcol, vrow


def EMGmm(X, gmm, threshold=1e-6, maxIter=1000000, psi=None, covType="full"):
    log_likelihood = []
    iteration = 0
    newGMM = []
    for _ in range(maxIter):
        responsability, logLikelihoodReturned = E_Step(X, gmm)
        gmm = M_Step(X, responsability, gmm, psi, covType)
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


def M_Step(X, responsability, gmm, psi=None, covType="full"):
    newGMM = []
    for gIndex in range(len(gmm)):
        gamma = responsability[gIndex]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        S = (vrow(gamma) * X) @ X.T
        mu = F / Z
        C = S / Z - mu @ mu.T
        w = Z / X.shape[1]
        if covType == "diagonal":
            C = C * np.eye(X.shape[0])
        newGMM.append((w, mu, C))

    if covType == "tied":
        CTied = 0
        for gIndex in range(len(gmm)):
            wg, mug, Cg = newGMM[gIndex]
            CTied += wg * Cg
        newGMM = [(w, mu, CTied) for w, mu, C in newGMM]

    if psi is not None:
        newGMM = [(w, mu, smoothCovariance(C, psi)) for w, mu, C in newGMM]

    return newGMM


def smoothCovariance(cov, psi):
    U, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    return U @ (vcol(s) * U.T)
