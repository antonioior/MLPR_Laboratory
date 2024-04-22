import numpy as np


def vcol(data):
    return data.reshape(data.shape[0], 1)


def vrow(data):
    return data.reshape(1, data.shape[0])


def logpdf_GAU_ND(X, mu, C):
    Y = []
    # Number of features
    N = X.shape[0]
    # Iter for each input data
    for x in X.T:
        x = vcol(x)
        const = N*np.log(2*np.pi)
        logC = np.linalg.slogdet(C)[1]
        mult = np.dot(np.dot((x-mu).T, np.linalg.inv(C)), (x-mu))[0, 0]
        Y.append(-0.5*(const + logC + mult))
    return np.array(Y)

def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()