import numpy as np

from load import load_gmm
from utils import logpdf_GMM

if __name__ == "__main__":
    printResult = True

    # WITH 4D DATA
    data = np.load("../Data/GMM_data_4D.npy")
    gmm = load_gmm("../Data/GMM_4D_3G_init.json")
    logdensResult4D = np.load("../Data/GMM_4D_3G_init_ll.npy")
    logdens4D = logpdf_GMM(data, gmm)

    # WITH 1D DATA
    data = np.load("../Data/GMM_data_1D.npy")
    gmm = load_gmm("../Data/GMM_1D_3G_init.json")
    logdensResult1D = np.load("../Data/GMM_1D_3G_init_ll.npy")
    logdens1D = logpdf_GMM(data, gmm)

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
