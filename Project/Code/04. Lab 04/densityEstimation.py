import matplotlib.pyplot as plt
import numpy as np
from utils import compute_mu_C, vrow, loglikelihood, logpdf_GAU_ND
from graph import createGraphicDensityEstimation


def densityEstimation(D, L, createGraph=False, printResults=False, comment=""):
    x = np.linspace(-5, 5, 1000)
    plt.figure("Density Estimation", figsize=(20, 10))
    if printResults:
        print("DENSITY ESTIMATION - RESULT" + " " + comment)
    for i in range(D.shape[0]):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        muFalse, CFalse = compute_mu_C(np.array(D[i:i+1, L == 0]))
        yFalse = np.exp(logpdf_GAU_ND(vrow(x), muFalse, CFalse))

        muTrue, CTrue = compute_mu_C(np.array(D[i:i+1, L == 1]))
        yTrue = np.exp(logpdf_GAU_ND(vrow(x), muTrue, CTrue))

        llFalse = loglikelihood(vrow(x), muFalse, CFalse)
        llTrue = loglikelihood(vrow(x), muTrue, CTrue)

        if printResults:
            print("\tFeature ", i+1)
            print(f"\t\tll for false {llFalse:.3f}")
            print(f"\t\tll for true {llTrue:.3f}")

        if createGraph:
            plt.subplot(2, 3, i+1)
            createGraphicDensityEstimation(D0, D1, i, "feature " + str(i+1), x.ravel(), yFalse, yTrue)

    if createGraph:
        plt.show()
    plt.show()
