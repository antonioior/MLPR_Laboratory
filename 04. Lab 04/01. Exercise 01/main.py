import numpy as np
import matplotlib.pyplot as plt
import utils as ut


if __name__ == "__main__":
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.
    C = np.ones((1, 1)) * 2.
    plt.plot(XPlot.ravel(), np.exp(ut.logpdf_GAU_ND(ut.vrow(XPlot), m, C)))
    plt.show()
    pdfSol = np.load("../Solutions/llGAU.npy")
    pdfGau = ut.logpdf_GAU_ND(ut.vrow(XPlot), m, C)
    print(np.abs(pdfSol - pdfGau).max())

    XND = np.load("../Solutions/XND.npy")
    mu = np.load("../Solutions/muND.npy")
    C = np.load("../Solutions/CND.npy")
    pdfSol = np.load("../Solutions/llND.npy")
    pdfGau = ut.logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())

    # First Dataset
    mu = ut.vcol(XND.mean(axis=1))
    var = XND.var(axis=1)
    DC = XND - mu
    C = (DC.dot(DC.T)) / (XND.shape[1])
    ll = ut.loglikelihood(XND, mu, C)
    print("Mu", mu)
    print("C", C)
    print("ll", ll)

    # Second Dataset
    X1D = np.load("./Solutions/X1D.npy")
    mu = ut.vcol(X1D.mean(axis=1))
    DC = X1D - mu
    C = (DC.dot(DC.T)) / (X1D.shape[1])
    ll = ut.loglikelihood(X1D, mu, C)
    print("Mu second dataset", mu)
    print("C second dataset", C)
    print("ll second dataset", ll)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(ut.logpdf_GAU_ND(ut.vrow(XPlot), mu, C)))
    plt.show()
