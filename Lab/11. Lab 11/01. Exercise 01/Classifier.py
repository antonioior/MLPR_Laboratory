import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF, bayesError
from graph import createBayesErrorPlots
from printValue import printEvaluationResult, printCalibrationResult, printKFoldResult, printScoreFusionResult
from utils import vcol, vrow


class Classifier:
    def __init__(self, system, scoreCalDat, labelsCalDat, scoreEvalDat, labelEvalDat, priorT):
        self.system = system
        self.scoreCalDat = scoreCalDat
        self.labelsCalDat = labelsCalDat
        self.scoreEvalDat = scoreEvalDat
        self.labelsEvalDat = labelEvalDat
        self.priorT = priorT
        self.minDCF = minDCF(self.scoreCalDat, self.labelsCalDat, self.priorT, 1, 1)
        self.actDCF = actDCF(self.scoreCalDat, self.labelsCalDat, self.priorT, 1, 1)
        self.minDCFE = minDCF(self.scoreEvalDat, self.labelsEvalDat, self.priorT, 1, 1)
        self.actDCFE = actDCF(self.scoreEvalDat, self.labelsEvalDat, self.priorT, 1, 1)
        self.SCAL, self.SVAL, self.LCAL, self.LVAL = 0, 0, 0, 0
        self.minDCFCalValRaw, self.actDCFCalValRaw = 0, 0
        self.minDCFCalValCal, self.actDCFCalValCal = 0, 0
        self.minDCFEvalRawScore, self.actDCFEvalRawScore = 0, 0
        self.minDCFEvalCalScore, self.actDCFEvalCalScore = 0, 0
        self.calibratedSVALK = []
        self.labelK = []
        self.minDCFKFoldCalValCal, self.actDCFKFoldCalValCal = 0, 0

        self.calibratedKEval = 0
        self.minDCFKFoldEvalCal, self.actDCFKFoldEvalCal = 0, 0
        self.fusedSVAL, self.minDCFFusionSingleFold, self.actDCFFusionSingleFold = 0, 0, 0
        self.minDCFFusionSingleFoldEval, self.actDCFFusionSingleFoldEval = 0, 0
        self.minDCFFusionKFold, self.actDCFFusionKFold = 0, 0
        self.fusedScoresKFold, self.fusedLabels = [], []

        self.fusedScoresKFoldEval, self.minDCFFusionKFoldEval, self.actDCFFusionKFoldEval = 0, 0, 0

    def BayesError(self, llr, LTE, xRange, yRange, colorActDCF, colorMinDCF, title, show, labelActDCF, labelMinDCF,
                   linestyleActDCF, linestyleMinDCF, llrOther=None, lteOther=None, labelActDCFOther="",
                   colorActDCFOther="b",
                   linestyleActDCFOther="-"):
        effPriorLogOdds, actDCFBayesError, minDCFBayesError = bayesError(
            llr=llr,
            LTE=LTE,
            lineLeft=xRange[0],
            lineRight=xRange[1],
        )
        actDCFBayesErrorOther = None
        if llrOther is not None:
            _, actDCFBayesErrorOther, _ = bayesError(
                llr=llrOther,
                LTE=lteOther if lteOther is not None else LTE,
                lineLeft=xRange[0],
                lineRight=xRange[1],
            )

        createBayesErrorPlots(
            x=effPriorLogOdds,
            yActDCF=actDCFBayesError,
            yMinDCF=minDCFBayesError,
            xlim=xRange,
            ylim=yRange,
            colorActDCF=colorActDCF,
            colorMinDCF=colorMinDCF,
            title=title,
            show=show,
            labelActDCF=labelActDCF,
            labelMinDCF=labelMinDCF,
            linestyleActDCF=linestyleActDCF,
            linestyleMinDCF=linestyleMinDCF,
            yActDCFOther=actDCFBayesErrorOther,
            labelActDCFOther=labelActDCFOther,
            colorActDCFOther=colorActDCFOther,
            linestyleActDCFOther=linestyleActDCFOther
        )

    def splitScores(self):
        self.SCAL, self.SVAL = self.scoreCalDat[::3], np.hstack([self.scoreCalDat[1::3], self.scoreCalDat[2::3]])
        self.LCAL, self.LVAL = self.labelsCalDat[::3], np.hstack([self.labelsCalDat[1::3], self.labelsCalDat[2::3]])
        self.minDCFCalValRaw = minDCF(self.SVAL, self.LVAL, self.priorT, 1, 1)
        self.actDCFCalValRaw = actDCF(self.SVAL, self.LVAL, self.priorT, 1, 1)

        self.minDCFEvalRawScore = minDCF(self.scoreEvalDat, self.labelsEvalDat, self.priorT, 1, 1)
        self.actDCFEvalRawScore = actDCF(self.scoreEvalDat, self.labelsEvalDat, self.priorT, 1, 1)

    def calibration(self):
        l = 0
        w, b = training(vrow(self.SCAL), self.LCAL, l, self.priorT)
        self.calibrated_SVAL = (w.T @ vrow(self.SVAL) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
        self.minDCFCalValCal = minDCF(self.calibrated_SVAL, self.LVAL, self.priorT, 1, 1)
        self.actDCFCalValCal = actDCF(self.calibrated_SVAL, self.LVAL, self.priorT, 1, 1)

        self.calibrated_SVALEval = (w.T @ vrow(self.scoreEvalDat) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
        self.minDCFEvalCalScore = minDCF(self.calibrated_SVALEval, self.labelsEvalDat, self.priorT, 1, 1)
        self.actDCFEvalCalScore = actDCF(self.calibrated_SVALEval, self.labelsEvalDat, self.priorT, 1, 1)

    def KFold(self, K):
        for i in range(K):
            SCAL, SVAL = np.hstack([self.scoreCalDat[jdx::K] for jdx in range(K) if jdx != i]), self.scoreCalDat[i::K]
            LCAL, LVAL = np.hstack([self.labelsCalDat[jdx::K] for jdx in range(K) if jdx != i]), self.labelsCalDat[i::K]
            w, b = training(vrow(SCAL), LCAL, 0, self.priorT)
            calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
            self.calibratedSVALK.append(calibrated_SVAL)
            self.labelK.append(LVAL)

        self.calibratedSVALK = np.hstack(self.calibratedSVALK)
        self.labelK = np.hstack(self.labelK)
        self.minDCFKFoldCalValCal = minDCF(self.calibratedSVALK, self.labelK, self.priorT, 1, 1)
        self.actDCFKFoldCalValCal = actDCF(self.calibratedSVALK, self.labelK, self.priorT, 1, 1)

        w, b = training(vrow(self.scoreCalDat), self.labelsCalDat, 0, self.priorT)
        self.calibratedKEval = (w.T @ vrow(self.scoreEvalDat) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
        self.minDCFKFoldEvalCal = minDCF(self.calibratedKEval, self.labelsEvalDat, self.priorT, 1, 1)
        self.actDCFKFoldEvalCal = actDCF(self.calibratedKEval, self.labelsEvalDat, self.priorT, 1, 1)

    def scoreFusion(self, other, K):

        # Single fold
        w, b = training(np.vstack([self.SCAL, other.SCAL]), self.LCAL, 0, self.priorT)
        self.fusedSVAL = (
                w.T @ np.vstack([self.SVAL, other.SVAL]) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
        self.minDCFFusionSingleFold = minDCF(self.fusedSVAL, self.LVAL, self.priorT, 1, 1)
        self.actDCFFusionSingleFold = actDCF(self.fusedSVAL, self.LVAL, self.priorT, 1, 1)

        self.fusedSVALEval = (w.T @ np.vstack([self.scoreEvalDat, other.scoreEvalDat]) + b - np.log(
            self.priorT / (1 - self.priorT))).ravel()
        self.minDCFFusionSingleFoldEval = minDCF(self.fusedSVALEval, self.labelsEvalDat, self.priorT, 1, 1)
        self.actDCFFusionSingleFoldEval = actDCF(self.fusedSVALEval, self.labelsEvalDat, self.priorT, 1, 1)

        # K-Fold
        for i in range(K):
            SCAL1, SVAL1 = np.hstack([self.scoreCalDat[jdx::K] for jdx in range(K) if jdx != i]), self.scoreCalDat[i::K]
            SCAL2, SVAL2 = np.hstack([other.scoreCalDat[jdx::K] for jdx in range(K) if jdx != i]), other.scoreCalDat[
                                                                                                   i::K]
            LCAL, LVAL = np.hstack([self.labelsCalDat[jdx::K] for jdx in range(K) if jdx != i]), self.labelsCalDat[i::K]
            SCAL = np.vstack([SCAL1, SCAL2])
            w, b = training(SCAL, LCAL, 0, self.priorT)
            SVAL = np.vstack([SVAL1, SVAL2])
            calibrated_SVAL = (w.T @ SVAL + b - np.log(self.priorT / (1 - self.priorT))).ravel()
            self.fusedScoresKFold.append(calibrated_SVAL)
            self.fusedLabels.append(LVAL)

        self.fusedScoresKFold = np.hstack(self.fusedScoresKFold)
        self.fusedLabels = np.hstack(self.fusedLabels)
        self.minDCFFusionKFold = minDCF(self.fusedScoresKFold, self.fusedLabels, self.priorT, 1, 1)
        self.actDCFFusionKFold = actDCF(self.fusedScoresKFold, self.fusedLabels, self.priorT, 1, 1)

        w, b = training(np.vstack([self.scoreCalDat, other.scoreCalDat]), self.labelsCalDat, 0, self.priorT)
        self.fusedScoresKFoldEval = (w.T @ np.vstack([self.scoreEvalDat, other.scoreEvalDat]) + b - np.log(
            self.priorT / (1 - self.priorT))).ravel()
        self.minDCFFusionKFoldEval = minDCF(self.fusedScoresKFoldEval, self.labelsEvalDat, self.priorT, 1, 1)
        self.actDCFFusionKFoldEval = actDCF(self.fusedScoresKFoldEval, self.labelsEvalDat, self.priorT, 1, 1)

    def printEvaluation(self, xRange, yRange, colorActDCF="b", colorMinDCF="b"):
        printEvaluationResult(self, xRange, yRange, colorActDCF, colorMinDCF)

    def printCalibration(self, xRange, yRange, colorActDCF="b", colorMinDCF="b", colorACTDCFOther="b", numRow=1,
                         numCol=1, startIndex=1):
        printCalibrationResult(self, xRange, yRange, colorActDCF, colorMinDCF, colorACTDCFOther, numRow,
                               numCol, startIndex)

    def printKFold(self, xRange, yRange, colorActDCF="b", colorMinDCF="b", colorACTDCFOther="b", numRow=1,
                   numCol=1, startIndex=1):
        printKFoldResult(self, xRange, yRange, colorActDCF, colorMinDCF, colorACTDCFOther, numRow,
                         numCol, startIndex)

    def printScoreFusion(self, other, xRange, yRange, fusion, colorActDCF="b", colorMinDCF="b", colorACTDCFOther="b",
                         numRow=1,
                         numCol=1, startIndex=1):
        printScoreFusionResult(self, other, xRange, yRange, colorActDCF, colorMinDCF, colorACTDCFOther, numRow,
                               numCol, startIndex)


def training(SCAL, LCAL, l, priorT):
    ZTR = LCAL * 2.0 - 1.0  # We do it outside the objective function, since we only need to do it once

    wTar = priorT / (ZTR > 0).sum()  # Compute the weights for the two classes
    wNon = (1 - priorT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):  # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]

        s = np.dot(vcol(w).T, SCAL).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTar  # Apply the weights to the loss computations
        loss[ZTR < 0] *= wNon

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTar  # Apply the weights to the gradient computations
        G[ZTR < 0] *= wNon

        GW = (vrow(G) * SCAL).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w) ** 2, np.hstack([GW, np.array(Gb)])

    vf = sp.fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(SCAL.shape[0] + 1))[0]
    return vf[:-1], vf[-1]
