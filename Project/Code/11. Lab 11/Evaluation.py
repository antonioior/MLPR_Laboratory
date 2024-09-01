from DCF import minDCF, actDCF

from PriorWeightedBinLogReg import priorWeightedLogClass
from utils import vrow
import numpy as np
import scipy.optimize as sp
from printValue import printData


def evaluation(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, prior_Cal, printResult=False):
    evaluationQLR(DVAL, LVAL, qlr, evalData, evalLabels, pT, prior_Cal, printResult)
    evaluationSVM(DVAL, LVAL, svm, evalData, evalLabels, pT, prior_Cal, printResult)
    evaluationGMM(DVAL, LVAL, gmm, evalData, evalLabels, pT, prior_Cal, printResult)
    evaluationFusion(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, prior_Cal, printResult)


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


def evaluationFusion(DVAL, LVAL, qlr, svm, gmm, evalData, evalLabels, pT, priorCal, printResult):
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
        printData(minDCFEval, actDCFEval, minDCFEvalCal, actDCFEvalCal, fusedScore, evalLabels,
                  calibratedFusion_eval, evalLabels, "Fusion - calibration evaluation", "red")
