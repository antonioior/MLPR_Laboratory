import numpy as np

from EMGMM import EMGmm, smoothCovariance
from utils import compute_mu_C


def LBGAlgorithm(X, alpha, numComponent, psi=None, covType="full"):
    mu, C = compute_mu_C(X)

    if covType == "diagonal":
        C = C * np.eye(X.shape[0])

    if psi is not None:
        gmm = [(1.0, mu, smoothCovariance(C, psi))]
    else:
        gmm = [(1.0, mu, C)]

    while len(gmm) < numComponent:
        gmm = split_GMM(gmm, alpha)
        _, gmm = EMGmm(X, gmm, psi=psi, covType=covType)
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
