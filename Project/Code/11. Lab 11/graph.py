# LAB 02
# Functoin to create the graphics
import matplotlib.pyplot as plt


def createGraphic(data, labels, properties, initialProperty, finalProperty, comment):
    data0 = data[:, labels == 0]
    data1 = data[:, labels == 1]
    plt.figure("Graphics of the properties " + properties[initialProperty] + " and " + properties[finalProperty - 1] +
               " " + comment, figsize=(10, 10))
    count = 0
    for i in range(initialProperty, finalProperty):
        for j in range(initialProperty, finalProperty):
            count += 1
            if i == j:
                plt.subplot(finalProperty - initialProperty, finalProperty - initialProperty, count)
                hist(data0, data1, i, properties[i])
            else:
                plt.subplot(finalProperty - initialProperty, finalProperty - initialProperty, count)
                scatter(data0, data1, i, j, properties[i], properties[j])
    plt.tight_layout()


# Function called by createGraphic to create the histogram
def hist(D0, D1, property, features, title="", bins=10):
    plt.hist(D0[property, :], density=True, label='False', alpha=0.5, bins=bins)
    plt.hist(D1[property, :], density=True, label='True', alpha=0.5, bins=bins)
    plt.xlabel(features)
    plt.title(title)
    plt.legend()


# Function called by createGraphic to create the scatter plot
def scatter(D0, D1, propertyX, propertyY, featureX, featureY, alpha=0.5, title=""):
    plt.scatter(D0[propertyX, :], D0[propertyY, :], label='False', alpha=alpha)
    plt.scatter(D1[propertyX, :], D1[propertyY, :], label='True', alpha=alpha)
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.title(title)
    plt.legend()


# LAB 03
def representRatio(ratio):
    plt.figure("Percentage of variance")
    plt.plot(range(1, len(ratio) + 1), ratio, marker='o', linestyle='--')
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of variance")
    plt.title("Percentage of variance vs number of components")
    plt.show()


def createGraphicPCA_LDA(L, dataProjectedPCA, dataProjectedLDA, ):
    plt.figure("PCA vs LDA", figsize=(10, 10), dpi=200)
    plt.suptitle("PCA vs LDA")
    plt.subplot(2, 2, 1)
    scatter(dataProjectedPCA[:, L == 0], dataProjectedPCA[:, L == 1], 0, 1, "", "", title="PCA, 1st and 2nd direction")

    plt.subplot(2, 2, 2)
    scatter(dataProjectedLDA[:, L == 0], dataProjectedLDA[:, L == 1], 0, 1, "", "", title="LDA, 1st and 2nd direction")

    plt.subplot(2, 2, 3)
    hist(dataProjectedPCA[:, L == 0], dataProjectedPCA[:, L == 1], 0, "", title="PCA, 1st direction")

    plt.subplot(2, 2, 4)
    hist(dataProjectedLDA[:, L == 0], dataProjectedLDA[:, L == 1], 0, "", title="LDA, 1st direction")
    plt.gca().set_ylim([0, 0.6])
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
    plt.show()


def createGraphicTrainingLDA(projectedDataTraining, LTR, projectedDataValidation, LVAL, comment=""):
    plt.figure("Training vs Validation" + " " + comment)
    plt.suptitle("Training vs Validation" + " " + comment)
    plt.subplot(1, 2, 1)
    hist(projectedDataTraining[:, LTR == 0], projectedDataTraining[:, LTR == 1], 0, "", title="Training")
    plt.gca().set_ylim([0, 0.5])
    plt.subplot(1, 2, 2)
    hist(projectedDataValidation[:, LVAL == 0], projectedDataValidation[:, LVAL == 1], 0, "", title="Validation")
    plt.gca().set_ylim([0, 0.5])
    plt.show()


# LAB 04
def createGraphicDensityEstimation(D0, D1, property, feature, x, yFalse, yTrue):
    hist(D0, D1, property, feature, title="", bins=50)
    plt.plot(x, yFalse, label='False')
    plt.plot(x, yTrue, label='True')
    plt.legend()


# LAB 07
def createBayesErrorPlots(x, yActDCF, yMinDCF, xlim, ylim, colorActDCF, colorMinDCF, title, show=True,
                          labelActDCF="DCF", labelMinDCF="min DCF", linestyleActDCF="-", linestyleMinDCF="-"):
    plt.plot(x, yActDCF, label=fr"{labelActDCF}", color=colorActDCF, linestyle=linestyleActDCF)
    plt.plot(x, yMinDCF, label=fr"{labelMinDCF}", color=colorMinDCF, linestyle=linestyleMinDCF)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.title(title)
    if show:
        plt.show()


# LAB 08
def createMinDCFActDCFPlot(x, yMinDCF, yActDCF, colorMinDCF, colorActDCF, title="",
                           togheter=False, logScale=False, xLabel="lambda"):
    if togheter:
        plotDCF(x, yMinDCF, xLabel, "", "min DCF", colorMinDCF, title, logScale, False)
        plotDCF(x, yActDCF, xLabel, "", "act DCF", colorActDCF, title, logScale, False)
        plt.show()
    else:
        plotDCF(x, yMinDCF, xLabel, "minDCF value", "min DCF", colorMinDCF, title, logScale, True)
        plotDCF(x, yActDCF, xLabel, "acfDCF value", "act DCF", colorActDCF, title, logScale, True)


def plotDCF(x, y, xLabel, yLabel, label, color, title, logScale, show=False):
    plt.plot(x, y, label=label, color=color)
    plt.scatter(x, y, color=color)
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if logScale:
        plt.xscale('log', base=10)
    if show:
        plt.show()
