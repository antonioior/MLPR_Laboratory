import numpy as np

from Classifier import Classifier

if __name__ == "__main__":
    printResult = True
    scoreSys1 = np.load("../Data/scores_1.npy")
    scoreSys2 = np.load("../Data/scores_2.npy")
    label = np.load("../Data/labels.npy")
    priorT = 0.2

    system1 = Classifier("System 1", scoreSys1, label, priorT)
    system2 = Classifier("System 2", scoreSys2, label, priorT)

    if printResult:
        xRange = [-3, 3]
        yRange = [0, 1.3]
        system1.print()
        system2.print()
        system1.BayesError(xRange=xRange, yRange=yRange, colorActDCF="b", colorMinDCF="b", title="", show=False,
                           labelActDCF=f"actDCF ({system1.system})", labelMinDCF=f"minDCF ({system1.system})",
                           linestyleActDCF="-", linestyleMinDCF="--")
        system2.BayesError(xRange=xRange, yRange=yRange, colorActDCF="orange", colorMinDCF="orange", title="",
                           show=True,
                           labelActDCF=f"actDCF ({system2.system})", labelMinDCF=f"minDCF ({system2.system})",
                           linestyleActDCF="-", linestyleMinDCF="--")
