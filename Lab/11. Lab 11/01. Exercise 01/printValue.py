import matplotlib.pyplot as plt


def printEvaluationResult(classifier, xRange, yRange, colorActDCF, colorMinDCF):
    print(f"\t{classifier.system.upper()}")
    print(f"\t\tminDCF: {classifier.minDCF:.3f}")
    print(f"\t\tactDCF: {classifier.actDCF:.3f}")

    classifier.BayesError(llr=classifier.scoreCalDat, LTE=classifier.labelsCalDat, xRange=xRange, yRange=yRange,
                          colorActDCF=colorActDCF, colorMinDCF=colorMinDCF, title="Raw Scores",
                          show=False,
                          labelActDCF=f"actDCF ({classifier.system})", labelMinDCF=f"minDCF ({classifier.system})",
                          linestyleActDCF="-", linestyleMinDCF="--")


def printCalibrationResult(classifier, xRange, yRange, colorActDCF, colorMinDCF, colorACTDCFOther, numRow,
                           numCol, startIndex):
    print(f"\t{classifier.system.upper()}")
    print(f"\t\tCalibration validation dataset")
    printMinAndActDCF("Raw scores", classifier.minDCFCalValRaw, classifier.actDCFCalValRaw)
    plt.subplot(numRow, numCol, startIndex)
    classifier.BayesError(llr=classifier.SVAL, LTE=classifier.LVAL, xRange=xRange, yRange=yRange,
                          colorActDCF=colorActDCF,
                          colorMinDCF=colorMinDCF,
                          title=classifier.system.upper() + " Calibration validation, original raw scores",
                          show=False,
                          labelActDCF=f"actDCF", labelMinDCF=f"minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    printMinAndActDCF("Calibrated scores", classifier.minDCFCalValCal, classifier.actDCFCalValCal)
    plt.subplot(numRow, numCol, startIndex + 1)
    classifier.BayesError(llr=classifier.calibrated_SVAL, LTE=classifier.LVAL, xRange=xRange, yRange=yRange,
                          colorActDCF=colorActDCF,
                          colorMinDCF=colorMinDCF,
                          title=classifier.system.upper() + " Calibration validation, calibrated scores",
                          show=False,
                          labelActDCF=f"actDCF (cal.)", labelMinDCF=f"minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--", llrOther=classifier.SVAL,
                          labelActDCFOther="actDCF (pre-cal.)", colorActDCFOther=colorACTDCFOther,
                          linestyleActDCFOther=":")
    print(f"\t\tEvaluation dataset")
    printMinAndActDCF("Raw scores", classifier.minDCFEvalRawScore, classifier.actDCFEvalRawScore)
    printMinAndActDCF("Calibrated scores", classifier.minDCFEvalCalScore, classifier.actDCFEvalCalScore)
    plt.subplot(numRow, numCol, startIndex + 2)
    classifier.BayesError(llr=classifier.calibrated_SVALEval, LTE=classifier.labelsEvalDat, xRange=xRange,
                          yRange=yRange,
                          colorActDCF=colorActDCF,
                          colorMinDCF=colorMinDCF,
                          title=classifier.system.upper() + " Evaluation set, calibrated scores",
                          show=False,
                          labelActDCF=f"actDCF (cal.)", labelMinDCF=f"minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--", llrOther=classifier.scoreEvalDat,
                          labelActDCFOther="actDCF (pre-cal.)", colorActDCFOther=colorACTDCFOther,
                          linestyleActDCFOther=":")


def printKFoldResult(classifier, xRange, yRange, colorActDCF, colorMinDCF, colorACTDCFOther, numRow,
                     numCol, startIndex):
    print(f"\t{classifier.system.upper()}")
    print(f"\t\tCalibration validation dataset")
    printMinAndActDCF("Raw scores", classifier.minDCF, classifier.actDCF)
    printMinAndActDCF("Calibrated scores", classifier.minDCFKFoldCalValCal, classifier.actDCFKFoldCalValCal)
    plt.subplot(numRow, numCol, startIndex)
    classifier.BayesError(llr=classifier.calibratedSVALK, LTE=classifier.labelK, xRange=xRange, yRange=yRange,
                          colorActDCF=colorActDCF,
                          colorMinDCF=colorMinDCF,
                          title=classifier.system.upper() + " Calibration validation, KFold of calibrated scores",
                          show=False,
                          labelActDCF=f"actDCF (cal.)", labelMinDCF=f"minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--", llrOther=classifier.scoreCalDat,
                          lteOther=classifier.labelsCalDat,
                          labelActDCFOther="actDCF (pre-cal.)", colorActDCFOther=colorACTDCFOther,
                          linestyleActDCFOther=":")

    print(f"\t\tEvaluation dataset")
    printMinAndActDCF("Raw scores", classifier.minDCFE, classifier.actDCFE)
    printMinAndActDCF("Calibrated scores", classifier.minDCFKFoldEvalCal, classifier.actDCFKFoldEvalCal)
    plt.subplot(numRow, numCol, startIndex + 1)
    classifier.BayesError(llr=classifier.calibratedKEval, LTE=classifier.labelsEvalDat, xRange=xRange,
                          yRange=yRange,
                          colorActDCF=colorActDCF,
                          colorMinDCF=colorMinDCF,
                          title=classifier.system.upper() + " Calibration validation, KFold of calibrated scores",
                          show=False,
                          labelActDCF=f"actDCF (cal.)", labelMinDCF=f"minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--", llrOther=classifier.scoreEvalDat,
                          lteOther=classifier.labelsEvalDat,
                          labelActDCFOther="actDCF (pre-cal.)", colorActDCFOther=colorACTDCFOther,
                          linestyleActDCFOther=":")


def printMinAndActDCF(subtitle, minDCF, actDCF):
    print(f"\t\t\t{subtitle}")
    print(f"\t\t\t\tminDCF: {minDCF:.3f}")
    print(f"\t\t\t\tactDCF: {actDCF:.3f}")
