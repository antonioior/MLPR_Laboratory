import numpy as np

from utils import logpdf_GMM


def EMGmm(X, gmm, psi=None, covType="full"):
    thNew = None
    thOld = None
    N = X.shape[1]
    D = X.shape[0]

    while thOld == None or thNew - thOld > 1e-6:  # finchè non diverge
        thOld = thNew
        logSj, logSjMarg = logpdf_GMM(X, gmm)
        thNew = np.sum(logSjMarg) / N

        P = np.exp(logSj - logSjMarg)  # Responsabilità che è uguale alla probabilita a posteriori

        if covType == 'diagonal':
            newGmm = []
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (gamma.reshape(1, -1) * X).sum(1)
                S = np.dot(X, (gamma.reshape(1, -1) * X).T)
                w = Z / N
                mu = (F / Z).reshape(-1, 1)
                sigma = S / Z - np.dot(mu, mu.T)
                sigma *= np.eye(sigma.shape[0])
                U, s, _ = np.linalg.svd(sigma)
                s[s < psi] = psi
                sigma = np.dot(U, s.reshape(-1, 1) * U.T)
                newGmm.append((w, mu, sigma))
            gmm = newGmm

        elif covType == 'tied':
            newGmm = []
            sigmaTied = np.zeros((D, D))
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (gamma.reshape(1, -1) * X).sum(1)
                S = np.dot(X, (gamma.reshape(1, -1) * X).T)
                w = Z / N
                mu = (F / Z).reshape(-1, 1)
                sigma = S / Z - np.dot(mu, mu.T)
                sigmaTied += Z * sigma
                newGmm.append((w, mu))
            gmm = newGmm
            sigmaTied /= N
            U, s, _ = np.linalg.svd(sigmaTied)
            s[s < psi] = psi
            sigmaTied = np.dot(U, s.reshape(-1, 1) * U.T)

            newGmm = []
            for i in range(len(gmm)):
                (w, mu) = gmm[i]
                newGmm.append((w, mu, sigmaTied))

            gmm = newGmm

        else:
            newGmm = []
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (gamma.reshape(1, -1) * X).sum(1)
                S = np.dot(X, (gamma.reshape(1, -1) * X).T)

                w = Z / N
                mu = (F / Z).reshape(-1, 1)
                sigma = S / Z - np.dot(mu, mu.T)
                U, s, _ = np.linalg.svd(sigma)
                s[s < psi] = psi
                sigma = np.dot(U, s.reshape(-1, 1) * U.T)
                newGmm.append((w, mu, sigma))
            gmm = newGmm

    return gmm, thNew
