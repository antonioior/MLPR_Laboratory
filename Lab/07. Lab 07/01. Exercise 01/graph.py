import matplotlib.pyplot as plt


def createRocCurve(x, y, xlim, ylim, xlabel, ylabel):
    plt.plot(x, y)
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='dashed')
    plt.show()


def createBayesErrorPlots(x, yDCF, yMinDCF, xlim, ylim, epsilonValue, colorDCF, colorMinDCF):
    plt.plot(x, yDCF, label=fr"DCF ($\epsilon$ = {epsilonValue})", color=colorDCF)
    plt.plot(x, yMinDCF, label=fr"min DCF ($\epsilon$ = {epsilonValue})", color=colorMinDCF)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
