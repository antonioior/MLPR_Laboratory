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


def printScoreFusionResult(classifier, other, xRange, yRange, colorActDCF, colorMinDCF, colorACTDCFOther, numRow,
                           numCol, startIndex):
    print(f"\tCalibration validation dataset")
    print(f"\t\tSINGLE-FOLD")
    printMinAndActDCF("SYSTEM 1 (CAL.)", classifier.minDCFCalValCal, classifier.actDCFCalValCal)
    printMinAndActDCF("SYSTEM 2 (CAL.)", other.minDCFCalValCal, other.actDCFCalValCal)
    printMinAndActDCF("FUSION", classifier.minDCFFusionSingleFold, classifier.actDCFFusionSingleFold)
    plt.subplot(numRow, numCol, startIndex)
    classifier.BayesError(llr=classifier.calibrated_SVAL, LTE=classifier.LVAL, xRange=xRange, yRange=yRange,
                          colorActDCF="b",
                          colorMinDCF="b",
                          title="Calibration validation, single fold",
                          show=False,
                          labelActDCF=f"S1 - actDCF", labelMinDCF=f"S1 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=other.calibrated_SVAL, LTE=other.LVAL, xRange=xRange, yRange=yRange,
                          colorActDCF="orange",
                          colorMinDCF="orange",
                          title="Calibration validation, single fold",
                          show=False,
                          labelActDCF=f"S2 - actDCF", labelMinDCF=f"S2 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=classifier.fusedSVAL, LTE=classifier.LVAL, xRange=xRange, yRange=yRange,
                          colorActDCF="green",
                          colorMinDCF="green",
                          title="Calibration validation, single fold",
                          show=False,
                          labelActDCF=f"Fusion - actDCF", labelMinDCF=f"Fusion - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")

    print(f"\t\tK-FOLD CALIBRATION")
    printMinAndActDCF("SYSTEM 1 (CAL.)", classifier.minDCFKFoldCalValCal, classifier.actDCFKFoldCalValCal)
    printMinAndActDCF("SYSTEM 2 (CAL.)", other.minDCFKFoldCalValCal, other.actDCFKFoldCalValCal)
    printMinAndActDCF("FUSION", classifier.minDCFFusionKFold, classifier.actDCFFusionKFold)
    plt.subplot(numRow, numCol, startIndex + 2)
    classifier.BayesError(llr=classifier.calibratedSVALK, LTE=classifier.labelK, xRange=xRange, yRange=yRange,
                          colorActDCF="b",
                          colorMinDCF="b",
                          title="Calibration validation, KFold",
                          show=False,
                          labelActDCF=f"S1 - actDCF", labelMinDCF=f"S2 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=other.calibratedSVALK, LTE=other.labelK, xRange=xRange, yRange=yRange,
                          colorActDCF="orange",
                          colorMinDCF="orange",
                          title="Calibration validation, KFold",
                          show=False,
                          labelActDCF=f"S2 - actDCF", labelMinDCF=f"S2 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=classifier.fusedScoresKFold, LTE=classifier.fusedLabels, xRange=xRange, yRange=yRange,
                          colorActDCF="green",
                          colorMinDCF="green",
                          title="Calibration validation, KFold",
                          show=False,
                          labelActDCF=f"S1 + S2 - KFold - actDCF(0.2)", labelMinDCF=f"S1 + S2 - KFold - minDCF(0.2)",
                          linestyleActDCF="-", linestyleMinDCF="--")

    print(f"\tEvaluation dataset")
    print(f"\t\tSINGLE FOLD")
    printMinAndActDCF("SYSTEM 1 (CAL.)", classifier.minDCFEvalCalScore, classifier.actDCFEvalCalScore)
    printMinAndActDCF("SYSTEM 2 (CAL.)", other.minDCFEvalCalScore, other.actDCFEvalCalScore)
    printMinAndActDCF("FSUSION", classifier.minDCFFusionSingleFoldEval, classifier.actDCFFusionSingleFoldEval)
    plt.subplot(numRow, numCol, startIndex + 1)
    classifier.BayesError(llr=classifier.calibrated_SVALEval, LTE=classifier.labelsEvalDat, xRange=xRange,
                          yRange=yRange,
                          colorActDCF="b",
                          colorMinDCF="b",
                          title="Evaluation, single fold",
                          show=False,
                          labelActDCF=f"S1 - actDCF", labelMinDCF=f"S1 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=other.calibrated_SVALEval, LTE=other.labelsEvalDat, xRange=xRange, yRange=yRange,
                          colorActDCF="orange",
                          colorMinDCF="orange",
                          title="Evaluation, single fold",
                          show=False,
                          labelActDCF=f"S2 - actDCF", labelMinDCF=f"S2 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=classifier.fusedSVALEval, LTE=classifier.labelsEvalDat, xRange=xRange, yRange=yRange,
                          colorActDCF="green",
                          colorMinDCF="green",
                          title="Evaluation, single fold",
                          show=False,
                          labelActDCF=f"Fusion - actDCF", labelMinDCF=f"Fusion - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    print(f"\t\tK-FOLD CALIBRATION")
    printMinAndActDCF("SYSTEM 1 (CAL.)", classifier.minDCFKFoldEvalCal, classifier.actDCFKFoldEvalCal)
    printMinAndActDCF("SYSTEM 2 (CAL.)", other.minDCFKFoldEvalCal, other.actDCFKFoldEvalCal)
    printMinAndActDCF("FUSION", classifier.minDCFFusionKFoldEval, classifier.actDCFFusionKFoldEval)
    plt.subplot(numRow, numCol, startIndex + 3)
    classifier.BayesError(llr=classifier.calibratedKEval, LTE=classifier.labelsEvalDat, xRange=xRange,
                          yRange=yRange,
                          colorActDCF="b",
                          colorMinDCF="b",
                          title="Evaluation, KFold",
                          show=False,
                          labelActDCF=f"S1 - actDCF", labelMinDCF=f"S1 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=other.calibratedKEval, LTE=other.labelsEvalDat, xRange=xRange, yRange=yRange,
                          colorActDCF="orange",
                          colorMinDCF="orange",
                          title="Evaluation, Kfold",
                          show=False,
                          labelActDCF=f"S2 - actDCF", labelMinDCF=f"S2 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")
    classifier.BayesError(llr=classifier.fusedScoresKFoldEval, LTE=classifier.labelsEvalDat, xRange=xRange,
                          yRange=yRange,
                          colorActDCF="green",
                          colorMinDCF="green",
                          title="Evaluation, Kfold",
                          show=False,
                          labelActDCF=f"S1 + S2 - actDCF", labelMinDCF=f"S1 + S2 - minDCF",
                          linestyleActDCF="-", linestyleMinDCF="--")


def printMinAndActDCF(subtitle, minDCF, actDCF):
    print(f"\t\t\t{subtitle}")
    print(f"\t\t\t\tminDCF: {minDCF:.3f}")
    print(f"\t\t\t\tactDCF: {actDCF:.3f}")
