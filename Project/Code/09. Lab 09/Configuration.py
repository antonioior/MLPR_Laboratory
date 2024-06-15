from graph import createBayesErrorPlots
from printValue import printConfusionMatrix
from utils import computePVAL, computeConfusionMatrix
from DCF import compute_minDCF_binary_fast, computeDCF, bayesError
import copy

def MainConfiguration(LTE, mapLlr, mapLlrPCA, printResults=False):
    configurations = {
        "config1": Configuration(0.5, 1, 1, LTE, mapLlr["MVG"], mapLlr["TIED"], mapLlr["NB"], False, []),
        "config2": Configuration(0.9, 1, 1, LTE, mapLlr["MVG"], mapLlr["TIED"], mapLlr["NB"], False, []),
        "config3": Configuration(0.1, 1, 1, LTE, mapLlr["MVG"], mapLlr["TIED"], mapLlr["NB"], False, []),
        "config4": Configuration(0.5, 1, 9, LTE, mapLlr["MVG"], mapLlr["TIED"], mapLlr["NB"], False, []),
        "config5": Configuration(0.5, 9, 1, LTE, mapLlr["MVG"], mapLlr["TIED"], mapLlr["NB"], False, []),
    }
    for key in configurations:
        configurations[key].computeConfusionMatrix()
        configurations[key].computeDCFDCFNormalizedAndMin()

    configurationEffectivePrior = {
        "config1": Configuration(0.1, 1, 1, LTE, mapLlrPCA["MVG"], mapLlrPCA["TIED"], mapLlrPCA["NB"], True,
                                 6),
        "config2": Configuration(0.5, 1, 1, LTE, mapLlrPCA["MVG"], mapLlrPCA["TIED"], mapLlrPCA["NB"], True,
                                 6),
        "config3": Configuration(0.9, 1, 1, LTE, mapLlrPCA["MVG"], mapLlrPCA["TIED"], mapLlrPCA["NB"], True,
                                 6),
    }

    for key in configurationEffectivePrior:
        configurationEffectivePrior[key].computeConfusionMatrix()
        configurationEffectivePrior[key].computeDCFDCFNormalizedAndMin()

    if printResults:
        for key in configurations:
            print(configurations[key])
        for key in configurationEffectivePrior:
            print(configurationEffectivePrior[key])
        configurationEffectivePrior["config1"].computeBayesErrorPlot()


class Configuration:
    def __init__(self, prior, Cfn, Cfp, LTE, llrMVG, llrTied, llrNB, PCA, m):
        self.prior = prior
        self.Cfn = Cfn
        self.Cfp = Cfp
        self.LTE = LTE
        self.PCA = PCA
        self.m = m
        self.classifier = {
            "MVG": {
                "llrs": copy.deepcopy(llrMVG)
            },
            "TIED": {
                "llrs": copy.deepcopy(llrTied)
            },
            "NB": {
                "llrs": copy.deepcopy(llrNB)
            }
        }

    def computeConfusionMatrix(self):
        if self.PCA == False:
            for key in self.classifier:
                pval = computePVAL(self.classifier[key]["llrs"], self.prior, self.Cfn, self.Cfp)
                self.classifier[key]["confusionMatrix"] = computeConfusionMatrix(pval, self.LTE)
        else:
            for key in self.classifier:
                for i in range(len(self.classifier[key]["llrs"])):
                    pval = computePVAL(self.classifier[key]["llrs"][i]["llr"], self.prior, self.Cfn, self.Cfp)
                    self.classifier[key]["llrs"][i]["confusionMatrix"] = computeConfusionMatrix(pval, self.LTE)

    def computeDCFDCFNormalizedAndMin(self):
        if self.PCA == False:
            for key in self.classifier:
                self.classifier[key]["DCF"] = computeDCF(
                    confusionMatrix=self.classifier[key]["confusionMatrix"],
                    pi=self.prior,
                    Cfn=self.Cfn,
                    Cfp=self.Cfp,
                    normalized=False)
                self.classifier[key]["DCFNormalized"] = computeDCF(
                    confusionMatrix=self.classifier[key]["confusionMatrix"],
                    pi=self.prior,
                    Cfn=self.Cfn,
                    Cfp=self.Cfp,
                    normalized=True)
                self.classifier[key]["DCFMin"] = compute_minDCF_binary_fast(
                    llr=self.classifier[key]["llrs"],
                    classLabels=self.LTE,
                    prior=self.prior,
                    Cfn=self.Cfn,
                    Cfp=self.Cfp,
                    returnThreshold=False)
        else:
            for key in self.classifier:
                for i in range(len(self.classifier[key]["llrs"])):
                    self.classifier[key]["llrs"][i]["DCF"] = computeDCF(
                        confusionMatrix=self.classifier[key]["llrs"][i]["confusionMatrix"],
                        pi=self.prior,
                        Cfn=self.Cfn,
                        Cfp=self.Cfp,
                        normalized=False)
                    self.classifier[key]["llrs"][i]["DCFNormalized"] = computeDCF(
                        confusionMatrix=self.classifier[key]["llrs"][i]["confusionMatrix"],
                        pi=self.prior,
                        Cfn=self.Cfn,
                        Cfp=self.Cfp,
                        normalized=True)
                    self.classifier[key]["llrs"][i]["DCFMin"] = compute_minDCF_binary_fast(
                        llr=self.classifier[key]["llrs"][i]["llr"],
                        classLabels=self.LTE,
                        prior=self.prior,
                        Cfn=self.Cfn,
                        Cfp=self.Cfp,
                        returnThreshold=False)

    def computeBayesErrorPlot(self):
        for key in self.classifier:
            best_m = mAssociatedToBestminDCF(self.classifier[key]["llrs"])
            effPriorLogOdds, dcfBayesError, minDCFBayesError = bayesError(
                llr=self.classifier[key]["llrs"][best_m - 1]["llr"],
                LTE=self.LTE,
                lineLeft=-4,
                lineRight=4
            )
            createBayesErrorPlots(effPriorLogOdds, dcfBayesError, minDCFBayesError, [-4, 4], [0, 1], "r", "b", key)

    def printConfusionMatrixSpecificClassifier(self, classifier):
        printConfusionMatrix(self.classifier[classifier]["confusionMatrix"])

    def printClassifier(self, classifier, numTab=2):
        print(f"{'\t' * (numTab - 1)}CLASSIFIER: {classifier}")
        if self.PCA == False:
            print(f"{'\t' * numTab}ConfusionMatrix:")
            self.printConfusionMatrixSpecificClassifier(classifier)
            print(f"{'\t' * numTab}DCF: {self.classifier[classifier]['DCF']: .5f}")
            print(f"{'\t' * numTab}DCF normalized: {self.classifier[classifier]['DCFNormalized']: .5f}")
            print(f"{'\t' * numTab}DCF min: {self.classifier[classifier]['DCFMin']: .5f}")

        else:
            for i in range(len(self.classifier[classifier]["llrs"])):
                print(f"{'\t' * numTab}ConfusionMatrix per m = {self.classifier[classifier]["llrs"][i]["m"]}:")
                printConfusionMatrix(self.classifier[classifier]["llrs"][i]["confusionMatrix"])
                print(f"{'\t' * numTab}DCF : {self.classifier[classifier]['llrs'][i]['DCF']: .5f}")
                print(f"{'\t' * numTab}DCF normalized : {self.classifier[classifier]['llrs'][i]['DCFNormalized']: .5f}")
                print(f"{'\t' * numTab}DCF min : {self.classifier[classifier]['llrs'][i]['DCFMin']: .5f}")
                print()

    def __str__(self):
        numTab = 2
        print(f"PCA: {self.PCA}")
        print(f"CONFIGURATION: ({self.prior}, {self.Cfn}, {self.Cfp})")
        self.printClassifier("MVG", numTab)
        self.printClassifier("TIED", numTab)
        self.printClassifier("NB", numTab)
        return ""


def mAssociatedToBestminDCF(dataClassifier):
    minDCFS = None
    mMinDCFS = None
    for i in range(len(dataClassifier)):
        if minDCFS is None or dataClassifier[i]["DCFMin"] < minDCFS:
            minDCFS = dataClassifier[i]["DCFMin"]
            mMinDCFS = dataClassifier[i]["m"]
    return mMinDCFS
