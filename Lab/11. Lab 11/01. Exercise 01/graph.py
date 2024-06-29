import matplotlib.pyplot as plt


# LAB 07
def createBayesErrorPlots(x, yActDCF, yMinDCF, xlim, ylim, colorActDCF, colorMinDCF, title, show=True,
                          labelActDCF="actDCF", labelMinDCF="minDCF", linestyleActDCF="-", linestyleMinDCF="-"):
    plt.plot(x, yActDCF, label=fr"{labelActDCF}", color=colorActDCF, linestyle=linestyleActDCF)
    plt.plot(x, yMinDCF, label=fr"{labelMinDCF}", color=colorMinDCF, linestyle=linestyleMinDCF)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.title(title)
    if show:
        plt.show()
