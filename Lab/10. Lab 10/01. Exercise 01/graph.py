import matplotlib.pyplot as plt


def plotEstimetedDensity(X1D, XPlot, likeLihood, title):
    plt.figure()
    plt.hist(X1D.ravel(), bins=25, density=True)
    plt.plot(XPlot.ravel(), likeLihood)
    plt.title(title)
    plt.show()
