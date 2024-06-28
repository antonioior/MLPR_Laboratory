from DCF import minDCF, actDCF
from LBG import LBGAlgorithm
from utils import logpdf_GMM


def GMM(DTR, LTR, DVAL, LVAL, printResults=False):
    covTypes = ["full", "diagonal"]
    components = [1, 2, 4, 8, 16, 32]
    psi = 0.01
    alpha = 0.1
    priorT = 0.1
    resultGMM = []
    count = 0
    for covType in covTypes:
        resultGMM.append({
            "covType": covType,
            "values": []
        })
        for component in components:
            gmm0 = LBGAlgorithm(DTR[:, LTR == 0], alpha, component, psi=psi, covType=covType)
            gmm1 = LBGAlgorithm(DTR[:, LTR == 1], alpha, component, psi=psi, covType=covType)

            SLLR = logpdf_GMM(DVAL, gmm1)[1] - logpdf_GMM(DVAL, gmm0)[1]
            minDCFValue = minDCF(SLLR, LVAL, priorT, 1.0, 1.0)
            actDCFValue = actDCF(SLLR, LVAL, priorT, 1.0, 1.0)
            resultGMM[count]["values"].append({
                "component": component,
                "minDCF": minDCFValue,
                "actDCF": actDCFValue
            })
        count += 1

    if printResults:
        print("GMM RESULTS")
        for result in resultGMM:
            print(f"\t{result['covType'].upper()}")
            for value in result["values"]:
                print(
                    f"\t\tcomponent = {value['component']}, minDCF = {value['minDCF']:.4f}, actDCF = {value['actDCF']:.4f}")
