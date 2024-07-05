import matplotlib.pyplot as plt


# LAB 07
def createBayesErrorPlots(x, yActDCF, yMinDCF, xlim, ylim, colorActDCF, colorMinDCF, title, show=True,
                          labelActDCF="actDCF", labelMinDCF="minDCF", linestyleActDCF="-", linestyleMinDCF="-",
                          yActDCFOther=None, labelActDCFOther="actDCF other", colorActDCFOther="b",
                          linestyleActDCFOther="-"):
    plt.plot(x, yActDCF, label=fr"{labelActDCF}", color=colorActDCF, linestyle=linestyleActDCF)
    plt.plot(x, yMinDCF, label=fr"{labelMinDCF}", color=colorMinDCF, linestyle=linestyleMinDCF)
    if yActDCFOther is not None:
        plt.plot(x, yActDCFOther, label=fr"{labelActDCFOther}", color=colorActDCFOther,
                 linestyle=linestyleActDCFOther)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.title(title, fontsize=10)
    if show:
        plt.show()
