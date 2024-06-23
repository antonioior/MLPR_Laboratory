import numpy as np

from EMGMM import EMGmm
from graph import plotEstimetedDensity
from load import load_gmm
from utils import logpdf_GMM
from utils import vrow

if __name__ == "__main__":
    printResult = True
    # GMM
    # WITH 4D DATA
    data4D = np.load("../Data/GMM_data_4D.npy")
    gmm4D = load_gmm("../Data/GMM_4D_3G_init.json")
    logdensResult4D = np.load("../Data/GMM_4D_3G_init_ll.npy")
    _, logdens4D = logpdf_GMM(data4D, gmm4D)

    # WITH 1D DATA
    data1D = np.load("../Data/GMM_data_1D.npy")
    gmm1D = load_gmm("../Data/GMM_1D_3G_init.json")
    logdensResult1D = np.load("../Data/GMM_1D_3G_init_ll.npy")
    _, logdens1D = logpdf_GMM(data1D, gmm1D)

    # GMM ESTIMATION
    logLikelihood4D, _ = EMGmm(data4D, gmm4D)
    logLikelihood1D, newGMM1D = EMGmm(data1D, gmm1D)

    if printResult:
        print("RESULT GMM")
        print("\t Logdens result prof 4D")
        print(logdensResult4D)
        print("\t Logdens calculated 4D")
        print(logdens4D)
        print("\t Logdens result prof 1D")
        print(logdensResult1D)
        print("\t Logdens calculated 1D")
        print(logdens1D)

        print("RESULT EM")
        print(f"\tAverage loglikelihood {sum(logLikelihood4D) / len(logLikelihood4D)}")
        XPlot = np.linspace(-10, 5, 1000)
        _, logDensXPlot = logpdf_GMM(vrow(XPlot), newGMM1D)
        likelihood = np.exp(logDensXPlot)
        plotEstimetedDensity(data1D, XPlot, likelihood)
