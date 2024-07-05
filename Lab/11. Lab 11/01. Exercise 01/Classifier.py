import numpy as np
import scipy.optimize as sp

from DCF import minDCF, actDCF, bayesError
from graph import createBayesErrorPlots
from utils import vcol, vrow


class Classifier:
    def __init__(self, system, score, labels, priorT):
        self.system = system
        self.score = score
        self.labels = labels
        self.priorT = priorT
        self.minDCF = minDCF(self.score, self.labels, self.priorT, 1, 1)
        self.actDCF = actDCF(self.score, self.labels, self.priorT, 1, 1)
        self.SCAL, self.SVAL, self.LCAL, self.LVAL = 0, 0, 0, 0
        self.minDCFCalValRaw, self.actDCFCalValRaw = 0, 0
        self.minDCFCalValCal, self.actDCFCalValCal = 0, 0

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
        l = 0
        ZTR = self.LCAL * 2.0 - 1.0  # We do it outside the objective function, since we only need to do it once

        wTar = self.priorT / (ZTR > 0).sum()  # Compute the weights for the two classes
        wNon = (1 - self.priorT) / (ZTR < 0).sum()

        def logreg_obj_with_grad(v):  # We compute both the objective and its gradient to speed up the optimization
            w = v[:-1]
            b = v[-1]

            s = np.dot(vcol(w).T, vrow(self.SCAL)).ravel() + b

            loss = np.logaddexp(0, -ZTR * s)
            loss[ZTR > 0] *= wTar  # Apply the weights to the loss computations
            loss[ZTR < 0] *= wNon

            G = -ZTR / (1.0 + np.exp(ZTR * s))
            G[ZTR > 0] *= wTar  # Apply the weights to the gradient computations
            G[ZTR < 0] *= wNon

            GW = (vrow(G) * vrow(self.SCAL)).sum(1) + l * w.ravel()
            Gb = G.sum()
            return loss.sum() + l / 2 * np.linalg.norm(w) ** 2, np.hstack([GW, np.array(Gb)])

        vf = sp.fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(vrow(self.SCAL).shape[0] + 1))[0]
        w, b = vf[:-1], vf[-1]
        calibrated_SVAL = (w.T @ vrow(self.SVAL) + b - np.log(self.priorT / (1 - self.priorT))).ravel()
        self.minDCFCalValCal = minDCF(calibrated_SVAL, self.LVAL, self.priorT, 1, 1)
        self.actDCFCalibrated = actDCF(calibrated_SVAL, self.LVAL, self.priorT, 1, 1)

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
        print(f"\t\t\t\tminDCF: {self.minDCFCalValCal:.3f}")
        print(f"\t\t\t\tactDCF: {self.actDCFCalValCal:.3f}")
        self.BayesError(xRange=xRange, yRange=yRange, colorActDCF=colorActDCF, colorMinDCF=colorMinDCF,
                        title=self.system.upper(),
                        show=True,
                        labelActDCF=f"actDCF", labelMinDCF=f"minDCF",
                        linestyleActDCF="-", linestyleMinDCF="--", fold=1)
