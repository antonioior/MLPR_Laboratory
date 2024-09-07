from GMM import GMMObject
from DCF import minDCF, actDCF
from utils import vrow
from PriorWeightedBinLogReg import priorWeightedLogClass
import scipy.optimize as sp
import numpy as np


def GMMOtherMethods(DTR, LTR, DVAL, LVAL, evalData, evalLabels, printResults=False):
    covTypes = ["full", "diagonal"]
    components = [1, 2, 4, 8, 16, 32]
    priorT = 0.1
    priorCal = 0.1
    psi = 0.01
    alpha = 0.1
    for covType in covTypes:
        for componentGMM0 in components:
            for componentGMM1 in components:
                gmm = GMMObject(DTR, LTR, componentGMM0, componentGMM1, alpha, psi, covType)
                _, _, _ = gmm.trainGMMReturnMinAndActDCF(DVAL, LVAL,
                                                         priorT)
                scoreGMM_eval = gmm.computeScore(evalData)
                minDCFEval = minDCF(scoreGMM_eval, evalLabels, priorT, 1, 1)
                actDCFEval = actDCF(scoreGMM_eval, evalLabels, priorT, 1, 1)

                logRegWeight = priorWeightedLogClass(vrow(gmm.computeScore(DVAL)), LVAL, 0, priorCal)
                vf = sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
                w, b = vf[:-1], vf[-1]
                calibratedScoreGMM_eval = (w.T @ vrow(scoreGMM_eval) + b - np.log(priorCal / (1 - priorCal))).ravel()
                minDCFEvalCal = minDCF(calibratedScoreGMM_eval, evalLabels, priorT, 1, 1)
                actDCFEvalCal = actDCF(calibratedScoreGMM_eval, evalLabels, priorT, 1, 1)

                if printResults:
                    print(f"\tcovType: {covType} - componentGMM0: {componentGMM0} - componentGMM1: {componentGMM1}")
                    print(f"\t\tminDCFWithoutCal: {minDCFEval:.4f}")
                    print(f"\t\tactDCFWithoutCal: {actDCFEval:.4f}")
                    print(f"\t\tminDCFEval: {minDCFEvalCal:.4f}")
                    print(f"\t\tactDCFEval: {actDCFEvalCal:.4f}")
