from GMM import GMMObject
from LBG import trainGMMCalibrationReturnMinAndActDCF
from Evaluation import evaluationGMM
from DCF import minDCF, actDCF


def GMMOtherMethods(DTR, LTR, DVAL, LVAL, evalData, evalLabels, printResults=False):
    covTypes = ["full", "diagonal"]
    components = [1, 2, 4, 8, 16, 32]
    priorT = 0.1
    priorCal = 0.1
    psi = 0.01
    alpha = 0.1
    K = 5
    for covType in covTypes:
        for componentGMM0 in components:
            for componentGMM1 in components:
                gmm = GMMObject(DTR, LTR, componentGMM0, componentGMM1, alpha, psi, covType)
                score, minDCFWithoutCal, actDCFWithoutCal = gmm.trainGMMReturnMinAndActDCF(DVAL, LVAL,
                                                                                           priorT)
                # minDCFKFold, actDCFKFold, llrK, labelK = trainGMMCalibrationReturnMinAndActDCF(K, priorCal, priorT,
                #                                                                                score, LVAL)
                calibratedScoreGMM_eval, evalLabels = evaluationGMM(DVAL, LVAL, gmm, evalData, evalLabels, priorT,
                                                                    priorCal, False)
                minDCFEvalCal = minDCF(calibratedScoreGMM_eval, evalLabels, priorT, 1, 1)
                actDCFEvalCal = actDCF(calibratedScoreGMM_eval, evalLabels, priorT, 1, 1)

                if printResults:
                    print(f"RESULT FOR CALIBRATION GMM")
                    print(f"\tcovType: {covType} - componentGMM0: {componentGMM0} - componentGMM1: {componentGMM1}")
                    print(f"\t\tminDCFWithoutCal: {minDCFWithoutCal}")
                    print(f"\t\tactDCFWithoutCal: {actDCFWithoutCal}")
                    print(f"\t\tminDCFEval: {minDCFEvalCal}")
                    print(f"\t\tactDCFEval: {actDCFEvalCal}")
