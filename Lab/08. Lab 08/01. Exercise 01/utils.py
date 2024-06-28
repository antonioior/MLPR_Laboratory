import numpy as np


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


def errorRate(DVAL, LVAL, vf):
    w, b = vf[:-1], vf[-1]
    sVal = np.dot(w.T, DVAL).ravel() + b
    Pval = (sVal > 0) * 1
    error = (Pval != LVAL).mean()
    return error, sVal


def computeOptimalBayesLlr(llr, prior, Cfn, Cfp):
    threshold = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    return np.int32(llr > threshold)


def computeConfusionMatrix(PVAL, LTE):
    num_classes = len(set(LTE))
    confusionMatrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(PVAL)):
        confusionMatrix[PVAL[i]][LTE[i]] += 1
    return confusionMatrix
