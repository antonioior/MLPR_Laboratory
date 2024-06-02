import matplotlib.pyplot as plt


def createRocCurve(x, y, xlim, ylim, xlabel, ylabel):
    plt.plot(x, y)
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='dashed')
    plt.show()


def createBayesErrorPlots(x, y1, y2, xlim, ylim):
    plt.plot(x, y1, label="DCF", color="r")
    plt.plot(x, y2, label="min DCF", color="b")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.show()
