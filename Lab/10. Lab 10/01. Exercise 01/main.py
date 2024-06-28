import numpy as np

from DCF import minDCF, actDCF
from EMGMM import EMGmm
from LBG import LBGAlgorithm
from graph import plotEstimetedDensity
from load import load_gmm, load_iris
from utils import logpdf_GMM
from utils import vrow, split_db_2to1, vcol

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

    # LBG ALGORITHM
    gmm4DLBG = LBGAlgorithm(data4D, 0.1, 4)
    gmm1DLBG = LBGAlgorithm(data1D, 0.1, 4)

    # GMM Classification IRIS
    D, L = load_iris()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    covTypes = ["full", "diagonal", "tied"]
    components = [1, 2, 4, 8, 16]
    psi = 0.01
    alpha = 0.1
    resultGMMIris = []
    for covType in covTypes:
        for component in components:
            gmm0 = LBGAlgorithm(DTR[:, LTR == 0], alpha, component, psi=psi, covType=covType)
            gmm1 = LBGAlgorithm(DTR[:, LTR == 1], alpha, component, psi=psi, covType=covType)
            gmm2 = LBGAlgorithm(DTR[:, LTR == 2], alpha, component, psi=psi, covType=covType)

            SVAL = []
            SVAL.append(logpdf_GMM(DVAL, gmm0)[1])
            SVAL.append(logpdf_GMM(DVAL, gmm1)[1])
            SVAL.append(logpdf_GMM(DVAL, gmm2)[1])
            SVAL = np.vstack(SVAL)
            SVAL += vcol(np.log(np.ones(3) / 3))
            PVAL = SVAL.argmax(0)
            errorRatePercentual = (LVAL != PVAL).sum() / LVAL.size * 100
            resultGMMIris.append({
                "covType": covType,
                "component": component,
                "errorRate": errorRatePercentual
            })

    # GMM CLASSIFICATION BINARY TASK
    D, L = np.load("../Data/ext_data_binary.npy"), np.load("../Data/ext_data_binary_labels.npy")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    resultGMMBinary = []
    for covType in covTypes:
        for component in components:
            gmm0 = LBGAlgorithm(DTR[:, LTR == 0], alpha, component, psi=psi, covType=covType)
            gmm1 = LBGAlgorithm(DTR[:, LTR == 1], alpha, component, psi=psi, covType=covType)

            SLLR = logpdf_GMM(DVAL, gmm1)[1] - logpdf_GMM(DVAL, gmm0)[1]
            minDCFValue = minDCF(SLLR, LVAL, 0.5, 1.0, 1.0)
            actDCFValue = actDCF(SLLR, LVAL, 0.5, 1.0, 1.0)
            resultGMMBinary.append({
                "covType": covType,
                "component": component,
                "minDCF": minDCFValue,
                "actDCF": actDCFValue
            })

    # PRINT RESULTS
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
        plotEstimetedDensity(data1D, XPlot, likelihood, "EM algorithm")

        print("RESULT LBG")
        print(f"\tAverage loglikelihood {logpdf_GMM(data4D, gmm4DLBG)[1].mean()}")
        _, logDensXPlot = logpdf_GMM(vrow(XPlot), gmm1DLBG)
        likelihood = np.exp(logDensXPlot)
        plotEstimetedDensity(data1D, XPlot, likelihood, "LBG algorithm")

        print("GMM CLASSIFICATION IRIS")
        for result in resultGMMIris:
            print(f"\t{result["covType"]}, component = {result["component"]} error rate = {result["errorRate"]:.1f}%")

        print("GMM CLASSIFICATION BINARY TASK")
        for result in resultGMMBinary:
            print(
                f"\t{result["covType"]}, component = {result["component"]} minDCF = {result["minDCF"]:.4f}, actDCF = {result["actDCF"]:.4f}")
