import itertools

import numpy as np
import scipy.special


def split_data(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
    return lTrain, lTest


def mcol(v):
    return v.reshape((v.size, 1))


def buildDictionary(ltercets):
    hDict = {}
    nWords = 0
    for tercet in ltercets:
        words = tercet.split()
        for w in words:
            if w not in hDict:
                hDict[w] = nWords
                nWords += 1
    return hDict


def S_estimateModel(hlTercets, eps=0.001):
    lTercetsAll = list(itertools.chain(*hlTercets.values()))
    hWordDict = buildDictionary(lTercetsAll)
    nWords = len(hWordDict)

    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = np.zeros(nWords) + eps

    for cls in hlTercets:
        lTercets = hlTercets[cls]
        for tercet in lTercets:
            words = tercet.split()
            for w in words:
                h_clsLogProb[cls][hWordDict[w]] += 1

    for cls in h_clsLogProb.keys():
        vOccurencies = h_clsLogProb[cls]
        vFrequencies = vOccurencies / vOccurencies.sum()
        vLogFrequencies = np.log(vFrequencies)
        h_clsLogProb[cls] = vLogFrequencies

    return h_clsLogProb, hWordDict


def compute_classPosteriors(S, logPrior=None):
    if logPrior is None:
        logPrior = np.log(np.ones(S.shape[0]) / float(S.shape[0]))
    J = S + mcol(logPrior)  # Compute joint probability
    ll = scipy.special.logsumexp(J, axis=0)  # Compute marginal likelihood log f(x)
    P = J - ll  # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return np.exp(P)


def S_tercet2occurrencies(tercet, hWordDict):
    v = np.zeros(len(hWordDict))
    for w in tercet.split():
        if w in hWordDict:  # We discard words that are not in the dictionary
            v[hWordDict[w]] += 1
    return mcol(v)


def S_compute_logLikelihoodMatrix(h_clsLogProb, hWordDict, lTercets, hCls2Idx=None):
    if hCls2Idx is None:
        hCls2Idx = {cls: idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    numClasses = len(h_clsLogProb)
    numWords = len(hWordDict)

    # We build the matrix of model parameters. Each row contains the model parameters for a class (the row index is given from hCls2Idx)
    MParameters = np.zeros((numClasses, numWords))
    for cls in h_clsLogProb:
        clsIdx = hCls2Idx[cls]
        MParameters[clsIdx, :] = h_clsLogProb[cls]

    SList = []
    for tercet in lTercets:
        v = S_tercet2occurrencies(tercet, hWordDict)
        STercet = np.dot(MParameters, v)
        SList.append(np.dot(MParameters, v))
    S = np.hstack(SList)
    return S


def compute_accuracy(P, L):
    PredictedLabel = np.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect) / float(NTotal)
