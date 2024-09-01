from DCF import minDCF, actDCF

from PriorWeightedBinLogReg import priorWeightedLogClass
from utils import vrow
import numpy as np
import scipy.optimize as sp
from printValue import printData
from graph import plotGraph


def evaluation(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, prior_Cal, printResult=False):
    scoreQLRCal, labelQLRCal = evaluationQLR(DVAL, LVAL, qlr, evalData, evalLabels, pT, prior_Cal, printResult)
    scoreSVMCal, labelSVMCal = evaluationSVM(DVAL, LVAL, svm, evalData, evalLabels, pT, prior_Cal, printResult)
    scoreGMMCal, labelGMMCal = evaluationGMM(DVAL, LVAL, gmm, evalData, evalLabels, pT, prior_Cal, printResult)
    evaluationFusion(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, prior_Cal, scoreQLRCal, labelQLRCal,
                     scoreSVMCal, labelSVMCal, scoreGMMCal, labelGMMCal, printResult)


def evaluationQLR(DVAL, LVAL, qlr, evalData, evalLabels, pT, priorCal, printResult=False):
    scoreQLR_eval = qlr.computeS(evalData)
    minDCFEval = minDCF(scoreQLR_eval, evalLabels, pT, 1, 1)
    actDCFEval = actDCF(scoreQLR_eval, evalLabels, pT, 1, 1)
    logRegWeight = priorWeightedLogClass(vrow(qlr.computeS(DVAL)), LVAL, 0, priorCal)
    vf = sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
    w, b = vf[:-1], vf[-1]
    calibratedScoreQLR_eval = (w.T @ vrow(scoreQLR_eval) + b - np.log(priorCal / (1 - priorCal))).ravel()
    minDCFEvalCal = minDCF(calibratedScoreQLR_eval, evalLabels, pT, 1, 1)
    actDCFEvalCal = actDCF(calibratedScoreQLR_eval, evalLabels, pT, 1, 1)

    if printResult:
        print("EVALUATION")
        print(f"\tQUADRATIC LOGISTIC REGRESSION")
        printData(minDCFEval, actDCFEval, minDCFEvalCal, actDCFEvalCal, scoreQLR_eval, evalLabels,
                  calibratedScoreQLR_eval, evalLabels, "QLR - calibration evaluation", "b")

    return calibratedScoreQLR_eval, evalLabels


def evaluationSVM(DVAL, LVAL, svm, evalData, evalLabels, pT, priorCal, printResult):
    scoreSVM_eval = svm.computeScore(svm.alphaStar, evalData)
    minDCFEval = minDCF(scoreSVM_eval, evalLabels, pT, 1, 1)
    actDCFEval = actDCF(scoreSVM_eval, evalLabels, pT, 1, 1)

    logRegWeight = priorWeightedLogClass(vrow(svm.computeScore(svm.alphaStar, DVAL)), LVAL, 0, priorCal)
    vf = sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
    w, b = vf[:-1], vf[-1]
    calibratedScoreSVM_eval = (w.T @ vrow(scoreSVM_eval) + b - np.log(priorCal / (1 - priorCal))).ravel()
    minDCFEvalCal = minDCF(calibratedScoreSVM_eval, evalLabels, pT, 1, 1)
    actDCFEvalCal = actDCF(calibratedScoreSVM_eval, evalLabels, pT, 1, 1)

    if printResult:
        print("\tSVM")
        printData(minDCFEval, actDCFEval, minDCFEvalCal, actDCFEvalCal, scoreSVM_eval, evalLabels,
                  calibratedScoreSVM_eval, evalLabels, "SVM - calibration evaluation",
                  "orange")

    return calibratedScoreSVM_eval, evalLabels


def evaluationGMM(DVAL, LVAL, gmm, evalData, evalLabels, pT, priorCal, printResult):
    scoreGMM_eval = gmm.computeScore(evalData)
    minDCFEval = minDCF(scoreGMM_eval, evalLabels, pT, 1, 1)
    actDCFEval = actDCF(scoreGMM_eval, evalLabels, pT, 1, 1)

    logRegWeight = priorWeightedLogClass(vrow(gmm.computeScore(DVAL)), LVAL, 0, priorCal)
    vf = sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
    w, b = vf[:-1], vf[-1]
    calibratedScoreGMM_eval = (w.T @ vrow(scoreGMM_eval) + b - np.log(priorCal / (1 - priorCal))).ravel()
    minDCFEvalCal = minDCF(calibratedScoreGMM_eval, evalLabels, pT, 1, 1)
    actDCFEvalCal = actDCF(calibratedScoreGMM_eval, evalLabels, pT, 1, 1)

    if printResult:
        print("\tGMM")
        printData(minDCFEval, actDCFEval, minDCFEvalCal, actDCFEvalCal, scoreGMM_eval, evalLabels,
                  calibratedScoreGMM_eval, evalLabels, "GMM - calibration evaluation",
                  "green")

    return calibratedScoreGMM_eval, evalLabels


def evaluationFusion(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, priorCal, scoreQLRCal, labelQLRCal,
                     scoreSVMCal, labelSVMCal, scoreGMMCal, labelGMMCal, printResult):
    scoreQLR = qlr.computeS(DVAL)
    screSVM = svm.computeScore(svm.alphaStar, DVAL)
    scoreGMM = gmm.computeScore(DVAL)
    fusion_score = np.vstack([scoreQLR, screSVM, scoreGMM])

    scoreQLR_eval = qlr.computeS(evalData)
    scoreSVM_eval = svm.computeScore(svm.alphaStar, evalData)
    scoreGMM_eval = gmm.computeScore(evalData)
    scoreEval = np.vstack([scoreQLR_eval, scoreSVM_eval, scoreGMM_eval])
    fusedScore = np.mean(scoreEval, axis=0)

    minDCFEval = minDCF(fusedScore, evalLabels, pT, 1, 1)
    actDCFEval = actDCF(fusedScore, evalLabels, pT, 1, 1)
    logRegWeight = priorWeightedLogClass(fusion_score, LVAL, 0, priorCal)
    vf = sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
    w, b = vf[:-1], vf[-1]
    calibratedFusion_eval = (w.T @ scoreEval + b - np.log(priorCal / (1 - priorCal))).ravel()
    minDCFEvalCal = minDCF(calibratedFusion_eval, evalLabels, pT, 1, 1)
    actDCFEvalCal = actDCF(calibratedFusion_eval, evalLabels, pT, 1, 1)

    if printResult:
        print("\tFUSION")
        print(f"\t\tminDCF: {minDCFEval:.4f}")
        print(f"\t\tactDCF: {actDCFEval:.4f}")
        print(f"\t\tminDCF - cal: {minDCFEvalCal:.4f}")
        print(f"\t\tactDCF - cal: {actDCFEvalCal:.4f}")
        plotGraph(scoreQLRCal, labelQLRCal, "b", f"Fusion - calibration priorCal = {priorCal}", False, " QLR - actDCF",
                  "QLR - minDCF", "-",
                  "--")
        plotGraph(scoreSVMCal, labelSVMCal, "orange", f"Fusion - calibration priorCal = {priorCal}", False,
                  "SVM - actDCF",
                  "SVM - minDCF", "-", "--")
        plotGraph(scoreGMMCal, labelGMMCal, "green", f"Fusion - calibration priorCal = {priorCal}", False,
                  "GMM - actDCF",
                  "GMM - minDCF", "-", "--")
        plotGraph(calibratedFusion_eval, evalLabels, "red", f"Fusion - calibration priorCal = {priorCal}", True,
                  "Fusion - actDCF",
                  "Fusion - minDCF", "-", "--")
