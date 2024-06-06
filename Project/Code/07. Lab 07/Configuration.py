from printValue import printConfusionMatrix
from utils import computePVAL, computeConfusionMatrix, computeDCF, compute_minDCF_binary_fast


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
                "llrs": llrMVG
            },
            "TIED": {
                "llrs": llrTied
            },
            "NB": {
                "llrs": llrNB
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
