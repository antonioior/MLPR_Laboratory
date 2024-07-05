import matplotlib.pyplot as plt
import numpy as np

from Classifier import Classifier

if __name__ == "__main__":
    printResult = True
    scoreSys1 = np.load("../Data/scores_1.npy")
    scoreSys2 = np.load("../Data/scores_2.npy")
    label = np.load("../Data/labels.npy")
    priorT = 0.2

    # EVALUATION
    system1 = Classifier("System 1", scoreSys1, label, priorT)
    system2 = Classifier("System 2", scoreSys2, label, priorT)

    # CALIBRATION
    system1.splitScores()
    system2.splitScores()

    system1.calibration()
    system2.calibration()

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
        system1.printCalibration(xRange, yRange, "b", "b")
        system2.printCalibration(xRange, yRange, "orange", "orange")
