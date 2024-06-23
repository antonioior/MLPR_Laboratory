import numpy as np

from EMGMM import EMGmm
from utils import compute_mu_C


def LBGAlgorithm(X, alpha, numComponent):
    mu, C = compute_mu_C(X)
    gmm = [(1.0, mu, C)]
    while len(gmm) < numComponent:
        gmm = split_GMM(gmm, alpha)
        _, gmm = EMGmm(X, gmm)
    return gmm


def split_GMM(gmm, alpha):
    gmmOut = []
    for gIndex in range(len(gmm)):
        w, mu, C = gmm[gIndex]
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmOut.append((w * 0.5, mu - d, C))
        gmmOut.append((w * 0.5, mu + d, C))
    return gmmOut
