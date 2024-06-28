import numpy as np

from utils import S_estimateModel, compute_classPosteriors, S_compute_logLikelihoodMatrix, compute_accuracy


def inferno_paradiso(lInf_train, lPar_train, lInf_evaluation, lPar_evaluation):
    hCls2Idx = {'inferno': 0, 'paradiso': 1}
    hlTercetsTrain = {
        'inferno': lInf_train,
        'paradiso': lPar_train
    }
    lTercetsEval = lInf_evaluation + lPar_evaluation
    S_model, S_wordDict = S_estimateModel(hlTercetsTrain, eps=0.001)
    S_predictions = compute_classPosteriors(
        S_compute_logLikelihoodMatrix(
            S_model,
            S_wordDict,
            lTercetsEval,
            hCls2Idx,
        ),
        np.log(np.array([1. / 2., 1. / 2.]))
    )

    labelsInf = np.zeros(len(lInf_evaluation))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = np.zeros(len(lPar_evaluation))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = np.hstack([labelsInf, labelsPar])
    return compute_accuracy(S_predictions, labelsEval)


def inferno_purgatorio(lInf_train, lPur_train, lInf_evaluation, lPur_evalution):
    hCls2Idx = {'inferno': 0, 'purgatorio': 1}
    hlTercetsTrain = {
        'inferno': lInf_train,
        'purgatorio': lPur_train
    }
    hlTercetsEval = lInf_evaluation + lPur_evalution
    model, wordDict = S_estimateModel(hlTercetsTrain, eps=0.001)
    predictions = compute_classPosteriors(
        S_compute_logLikelihoodMatrix(
            model,
            wordDict,
            hlTercetsEval,
            hCls2Idx,
        ),
        np.log(np.array([1. / 2., 1. / 2.]))
    )
    labelsInf = np.zeros(len(lInf_evaluation))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPure = np.zeros(len(lPur_evalution))
    labelsPure[:] = hCls2Idx['purgatorio']

    labelsEval = np.hstack([labelsInf, labelsPure])
    return compute_accuracy(predictions, labelsEval)


def purgatorio_paradiso(lPur_train, lPar_train, lPur_evaluation, lPar_evaluation):
    hCls2Idx = {'purgatorio': 0, 'paradiso': 1}
    hlTercetsTrain = {
        'purgatorio': lPur_train,
        'paradiso': lPar_train
    }
    hlTercetsEval = lPur_evaluation + lPar_evaluation
    model, wordDict = S_estimateModel(hlTercetsTrain, eps=0.001)
    predictions = compute_classPosteriors(
        S_compute_logLikelihoodMatrix(
            model,
            wordDict,
            hlTercetsEval,
            hCls2Idx,
        ),
        np.log(np.array([1. / 2., 1. / 2.]))
    )
    labelsPur = np.zeros(len(lPur_evaluation))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsPar = np.zeros(len(lPar_evaluation))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = np.hstack([labelsPur, labelsPar])
    return compute_accuracy(predictions, labelsEval)
