import numpy as np
import scipy as sp


def vcol(data):
    return data.reshape(data.shape[0], 1)


def vrow(data):
    return data.reshape(1, data.shape[0])


# LAB 5
def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu) * (P @ (x - mu))).sum(0)


# LAB 10
def logpdf_GMM(X, gmm):
    M = len(gmm)
    N = X.shape[1]
    S = np.zeros((M, N))
    for g in range(M):
        w, mu, C = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    logdens = sp.special.logsumexp(S, axis=0)
    return S, logdens
