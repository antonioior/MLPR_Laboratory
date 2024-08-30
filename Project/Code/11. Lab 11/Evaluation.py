from DCF import minDCF, actDCF

from PriorWeightedBinLogReg import priorWeightedLogClass
from utils import vrow
import numpy as np
import scipy.optimize as sp
from printValue import printData


def evaluation(DVAL, LVAL, qlr, evalData, evalLabels, pT, prior_Cal, printResult=False):
    evaluationQLR(DVAL, LVAL, qlr, evalData, evalLabels, pT, prior_Cal, printResult)


def evaluationQLR(DVAL, LVAL, qlr, evalData, evalLabels, pT, priorCal, printResult=False):
    scoreQLR_eval = qlr.computeS(evalData)
    minDCFEval = minDCF(scoreQLR_eval, evalLabels, pT, 1, 1)
    actDCFEval = actDCF(scoreQLR_eval, evalLabels, pT, 1, 1)
    print("minDCF:", minDCFEval)
    print("actDCF:", actDCFEval)
    logRegWeight = priorWeightedLogClass(vrow(qlr.computeS(DVAL)), LVAL, 0, priorCal)
    vf = \
        sp.fmin_l_bfgs_b(func=logRegWeight.logreg_obj, x0=np.zeros(logRegWeight.DTR.shape[0] + 1))[0]
    w, b = vf[:-1], vf[-1]
    calibratedScoreQLR_eval = (w.T @ vrow(scoreQLR_eval) + b - np.log(priorCal / (1 - priorCal))).ravel()
    minDCFEvalCal = minDCF(calibratedScoreQLR_eval, evalLabels, pT, 1, 1)
    actDCFEvalCal = actDCF(calibratedScoreQLR_eval, evalLabels, pT, 1, 1)
    print("minDCF calibrated:", minDCFEvalCal)
    print("actDCF calibrated:", actDCFEvalCal)
    print()
    if printResult:
        print("EVALUATION")
        print(f"\tEVALUATION QUADRATIC LOGISTIC REGRESSION")
        printData(minDCFEval, actDCFEval, minDCFEvalCal, actDCFEvalCal, scoreQLR_eval, evalLabels,
                  calibratedScoreQLR_eval, evalLabels, "QLR - calibration evaluation", "b")
