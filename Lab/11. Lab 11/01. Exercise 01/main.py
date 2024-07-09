import matplotlib.pyplot as plt
import numpy as np

from Classifier import Classifier

if __name__ == "__main__":
    printResult = True
    scoreSys1CalDat = np.load("../Data/scores_1.npy")
    scoreSys2CalDat = np.load("../Data/scores_2.npy")
    labelCalDat = np.load("../Data/labels.npy")

    scoreSys1EvalDat = np.load("../Data/eval_scores_1.npy")
    scoreSys2EvalDat = np.load("../Data/eval_scores_2.npy")
    labelEvalDat = np.load("../Data/eval_labels.npy")
    priorT = 0.2

    # EVALUATION
    system1 = Classifier("System 1", scoreSys1CalDat, labelCalDat, scoreSys1EvalDat, labelEvalDat, priorT)
    system2 = Classifier("System 2", scoreSys2CalDat, labelCalDat, scoreSys2EvalDat, labelEvalDat, priorT)

    # CALIBRATION - SINGLE FOLD
    system1.splitScores()
    system2.splitScores()

    system1.calibration()
    system2.calibration()

    # K - FOLD
    K = 5
    system1.KFold(K)
    system2.KFold(K)

    # Score-level fusion
    system1.scoreFusion(system2)

    if printResult:
        xRange = [-3, 3]

        # EVALUATION
        print("RAW SCORES")
        yRange = [0, 1.3]
        system1.printEvaluation(xRange, yRange, "b", "b")
        system2.printEvaluation(xRange, yRange, "orange", "orange")
        plt.show()

        # CALIBRATION
        print("CALIBRATION")
        yRange = [0, 0.8]

        plt.figure("SINGLE FOLD CALIBRATION", figsize=(15, 8), dpi=300)
        plt.suptitle("SINGLE FOLD CALIBRATION")
        numRow = 2
        numCol = 3
        startIndex = 1
        system1.printCalibration(xRange, yRange, "b", "b", "b", numRow, numCol, startIndex)
        startIndex = 4
        system2.printCalibration(xRange, yRange, "orange", "orange", "orange", numRow, numCol, startIndex)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tight_layout()
        plt.show()

        # K - FOLD
        print("K-FOLD CALIBRATION")
        plt.figure("K-FOLD CALIBRATION", figsize=(15, 10), dpi=300)
        plt.suptitle("K-FOLD CALIBRATION")
        numRow = 2
        numCol = 2
        startIndex = 1
        system1.printKFold(xRange, yRange, "b", "b", "b", numRow, numCol, startIndex)
        startIndex = 3
        system2.printKFold(xRange, yRange, "orange", "orange", "orange", numRow, numCol, startIndex)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tight_layout()
        plt.show()

        # Score-level fusion
        print("SCORE-LEVEL FUSION")
        plt.figure("SCORE-LEVEL FUSION", figsize=(15, 10), dpi=300)
        plt.suptitle("SCORE-LEVEL FUSION")
        numRow = 2
        numCol = 2
        startIndex = 1
        system1.printScoreFusion(system2, xRange, yRange, True, "b", "b", "b", numRow, numCol, startIndex)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tight_layout()
        plt.show()
