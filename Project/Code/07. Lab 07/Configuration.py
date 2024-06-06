from printValue import printConfusionMatrix
from utils import computePVAL, computeConfusionMatrix, computeDCF, compute_minDCF_binary_fast


class Configuration:
    def __init__(self, prior, Cfn, Cfp, LTE, llrMVG, llrTied, llrNB):
        self.prior = prior
        self.Cfn = Cfn
        self.Cfp = Cfp
        self.LTE = LTE
        self.classifier = {
            "MVG": {
                "llr": llrMVG
            },
            "TIED": {
                "llr": llrTied
            },
            "NB": {
                "llr": llrNB
            }
        }

    def computeConfusionMatrix(self):
        for key in self.classifier:
            pval = computePVAL(self.classifier[key]["llr"], self.prior, self.Cfn, self.Cfp)
            self.classifier[key]["confusionMatrix"] = computeConfusionMatrix(pval, self.LTE)

    def computeDCFDCFNormalizedAndMin(self):
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
                llr=self.classifier[key]["llr"],
                classLabels=self.LTE,
                prior=self.prior,
                Cfn=self.Cfn,
                Cfp=self.Cfp,
                returnThreshold=False)

    def printConfusionMatrixSpecificClassifier(self, classifier):
        printConfusionMatrix(self.classifier[classifier]["confusionMatrix"])

    def printClassifier(self, classifier, numTab=2):
        print(f"{'\t' * (numTab - 1)}CLASSIFIER: {classifier}")
        print(f"{'\t' * numTab}ConfusionMatrix:")
        self.printConfusionMatrixSpecificClassifier(classifier)
        print(f"{'\t' * numTab}DCF: {self.classifier[classifier]['DCF']: .5f}")
        print(f"{'\t' * numTab}DCF normalized: {self.classifier[classifier]['DCFNormalized']: .5f}")
        print(f"{'\t' * numTab}DCF min: {self.classifier[classifier]['DCFMin']: .5f}")

    def __str__(self):
        numTab = 2
        print(f"CONFIGURATION: ({self.prior}, {self.Cfn}, {self.Cfp})")
        self.printClassifier("MVG", numTab)
        self.printClassifier("TIED", numTab)
        self.printClassifier("NB", numTab)
        return ""
