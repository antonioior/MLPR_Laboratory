import matplotlib.pyplot as plt


def createRocCurve(x, y, xlim, ylim, xlabel, ylabel):
    plt.plot(x, y)
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='dashed')
    plt.show()
