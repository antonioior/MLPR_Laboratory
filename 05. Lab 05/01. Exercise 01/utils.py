import numpy as np
import scipy as sp


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


def vcol(data):
    return data.reshape(data.shape[0], 1)


def vrow(data):
    return data.reshape(1, data.shape[0])


def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu) * (P @ (x - mu))).sum(0)


def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = vrow(sp.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost


def calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults=False):
    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    difference = np.abs(LVAL - PVAL)
    numOfErr = sum(x != 0 for x in difference)

    if printResults:
        print("Error - RESULTS")
        print(f"    Threshold: {threshold}")
        print(f"    Real values:\n\t{LVAL}")
        print(f"    Predicted values:\n\t{PVAL}")
        print(f"    Difference:\n\t{difference}")
        print(f"    Number of errors:\n\t{numOfErr}")
        print(f"    Error rate: {(numOfErr / LVAL.shape[0]) * 100:.1f}%")


def mcol(mu, shape):
    return mu.reshape(shape, 1)


def projection(U, m):
    return U[:, ::-1][:, 0:m]
