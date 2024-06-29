from DCF import minDCF, actDCF, bayesError
from graph import createBayesErrorPlots


class Classifier:
    def __init__(self, system, score, labels, priorT):
        self.system = system
        self.score = score
        self.labels = labels
        self.priorT = priorT
        self.minDCF = minDCF(self.score, self.labels, self.priorT, 1, 1)
        self.actDCF = actDCF(self.score, self.labels, self.priorT, 1, 1)

    def print(self):
        print(f"{self.system.upper()}")
        print(f"\tminDCF: {self.minDCF:.3f}")
        print(f"\tactDCF: {self.actDCF:.3f}")

    def computeBayesError(self, xRange):
        self.effPriorLogOdds, self.actDCFBayesError, self.minDCFBayesError = bayesError(
            llr=self.score,
            LTE=self.labels,
            lineLeft=xRange[0],
            lineRight=xRange[1],
        )

    def BayesError(self, xRange, yRange, colorActDCF, colorMinDCF, title, show, labelActDCF, labelMinDCF,
                   linestyleActDCF, linestyleMinDCF):
        self.computeBayesError(xRange)
        createBayesErrorPlots(
            x=self.effPriorLogOdds,
            yActDCF=self.actDCFBayesError,
            yMinDCF=self.minDCFBayesError,
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
