import numpy as np

from DCF import minDCF, actDCF, bayesError
from graph import createBayesErrorPlots
from logRegClass import binaryLogisticRegression
from utils import vrow


class Classifier:
    def __init__(self, system, score, labels, priorT):
        self.system = system
        self.score = score
        self.labels = labels
        self.priorT = priorT
        self.minDCF = minDCF(self.score, self.labels, self.priorT, 1, 1)
        self.actDCF = actDCF(self.score, self.labels, self.priorT, 1, 1)

    def computeBayesError(self, xRange, fold=None):
        if fold is None:
            return bayesError(
                llr=self.score,
                LTE=self.labels,
                lineLeft=xRange[0],
                lineRight=xRange[1],
            )
        else:
            return bayesError(
                llr=self.SVAL,
                LTE=self.LVAL,
                lineLeft=xRange[0],
                lineRight=xRange[1],
            )

    def BayesError(self, xRange, yRange, colorActDCF, colorMinDCF, title, show, labelActDCF, labelMinDCF,
                   linestyleActDCF, linestyleMinDCF, fold=None):
        effPriorLogOdds, actDCFBayesError, minDCFBayesError = self.computeBayesError(xRange, fold)
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
        )

    def splitScores(self):
        self.SCAL, self.SVAL = self.score[::3], np.hstack([self.score[1::3], self.score[2::3]])
        self.LCAL, self.LVAL = self.labels[::3], np.hstack([self.labels[1::3], self.labels[2::3]])
        self.minDCFCalValRaw = minDCF(self.SVAL, self.LVAL, self.priorT, 1, 1)
        self.actDCFCalValRaw = actDCF(self.SVAL, self.LVAL, self.priorT, 1, 1)

    def calibration(self):
        self.minDCFCalibrated, self.actDCFCalibrated = binaryLogisticRegression(vrow(self.SCAL), self.LCAL,
                                                                                vrow(self.SVAL), self.LVAL, self.priorT,
                                                                                0,
                                                                                pEmp=False)

    def printEvaluation(self, xRange, yRange, colorActDCF="b", colorMinDCF="b"):
        print(f"\t{self.system.upper()}")
        print(f"\t\tminDCF: {self.minDCF:.3f}")
        print(f"\t\tactDCF: {self.actDCF:.3f}")

        self.BayesError(xRange=xRange, yRange=yRange, colorActDCF=colorActDCF, colorMinDCF=colorMinDCF, title="",
                        show=False,
                        labelActDCF=f"actDCF ({self.system})", labelMinDCF=f"minDCF ({self.system})",
                        linestyleActDCF="-", linestyleMinDCF="--")

    def printCalibration(self, xRange, yRange, colorActDCF="b", colorMinDCF="b"):
        print(f"\t{self.system.upper()}")
        print(f"\t\tCalibration validation dataset")

        print(f"\t\t\tRaw scores")
        print(f"\t\t\t\tminDCF: {self.minDCFCalValRaw:.3f}")
        print(f"\t\t\t\tactDCF: {self.actDCFCalValRaw:.3f}")

        print(f"\t\t\tCalibrated scores")
        print(f"\t\t\t\tminDCF: {self.minDCFCalibrated:.3f}")
        print(f"\t\t\t\tactDCF: {self.actDCFCalibrated:.3f}")
        self.BayesError(xRange=xRange, yRange=yRange, colorActDCF=colorActDCF, colorMinDCF=colorMinDCF,
                        title=self.system.upper(),
                        show=True,
                        labelActDCF=f"actDCF", labelMinDCF=f"minDCF",
                        linestyleActDCF="-", linestyleMinDCF="--", fold=1)
