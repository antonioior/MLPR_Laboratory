import numpy as np

from estimationBinary import inferno_paradiso, inferno_purgatorio, purgatorio_paradiso
from load import load_data
from utils import split_data, compute_classPosteriors, S_estimateModel, S_compute_logLikelihoodMatrix, compute_accuracy

if __name__ == "__main__":
    printResult = True

    lInf, lPur, lPar = load_data()
    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    hCls2Idx = {"inferno": 0, "purgatorio": 1, "paradiso": 2}
    hlTercetsTrain = {
        "inferno": lInf_train,
        "purgatorio": lPur_train,
        "paradiso": lPar_train
    }

    lTerctsEval = lInf_evaluation + lPur_evaluation + lPar_evaluation

    S_model, S_wordDict = S_estimateModel(hlTercetsTrain, eps=0.001)
    S_predictions = compute_classPosteriors(
        S_compute_logLikelihoodMatrix(S_model, S_wordDict, lTerctsEval, hCls2Idx),
        np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
    )

    labelsInf = np.zeros(len(lInf_evaluation))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = np.zeros(len(lPar_evaluation))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = np.zeros(len(lPur_evaluation))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsEval = np.hstack([labelsInf, labelsPur, labelsPar])
    multiclass_accuracy = compute_accuracy(S_predictions, labelsEval)

    ### Binary from multiclass scores [Optional, for the standard binary case see below] ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    lTercetsEval = lInf_evaluation + lPar_evaluation
    S = S_compute_logLikelihoodMatrix(S_model, S_wordDict, lTercetsEval, hCls2Idx=hCls2Idx)

    SBinary = np.vstack([S[0:1, :], S[2:3, :]])
    P = compute_classPosteriors(SBinary)
    labelsEval = np.hstack([labelsInf, labelsPar])
    # Since labelsPar == 2, but the row of Paradiso in SBinary has become row 1 (row 0 is Inferno), we have to modify the labels for paradise, otherwise the function compute_accuracy will not work
    labelsEval[labelsEval == 2] = 1
    binary_from_multiclass_accuracy = compute_accuracy(P, labelsEval)

    ### Binary ###
    accuracyInferno_Paradiso = inferno_paradiso(lInf_train, lPar_train, lInf_evaluation, lPar_evaluation)
    accuracyInferno_Purgatorio = inferno_purgatorio(lInf_train, lPur_train, lInf_evaluation, lPur_evaluation)
    accuracyPurgatorio_Paradiso = purgatorio_paradiso(lPur_train, lPar_train, lPur_evaluation, lPar_evaluation)

    if printResult:
        print(f"Multiclass - S2 - Accuracy: {multiclass_accuracy * 100: .2f} %")
        print(f"Binary (From multiclass) - S2 - Accuracy: {binary_from_multiclass_accuracy * 100: .2f} %")
        print(f"Binary [inferno vs paradiso] - S2 - Accuracy: {accuracyInferno_Paradiso * 100: .2f} %")
        print(f"Binary [inferno vs purgatorio] - S2 - Accuracy: {accuracyInferno_Purgatorio * 100: .2f} %")
        print(f"Binary [purgatorio vs paradiso] - S2 - Accuracy: {accuracyPurgatorio_Paradiso * 100: .2f} %")
