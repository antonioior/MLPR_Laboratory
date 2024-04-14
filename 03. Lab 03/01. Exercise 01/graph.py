#LAB 02
import matplotlib.pyplot as plt

def createGraphic(data, labels, properties, comment):
    data0 = data[:, labels == 0]
    data1 = data[:, labels == 1]
    data2 = data[:, labels == 2]
    for i in range(0, len(properties)):
        for j in range(0, len(properties)):
            if i == j:
                plt.figure(properties[i] + " " + comment)
                hist(data0, data1, data2, i, properties[i])
            else:
                plt.figure(properties[i] + " " + properties[j] + " " + comment)
                plot(data0, data1, data2, i, j, properties[i], properties[j])
    plt.show()


def hist(D0, D1, D2, property, features, title="", alpha = 0.5, bins = 10):
    plt.hist(D0[property,:], density=True, label='Setosa', alpha=alpha, bins = bins)
    plt.hist(D1[property,:], density=True, label='Versicolor', alpha=alpha, bins = bins)
    plt.hist(D2[property,:], density=True, label='Virginica', alpha=alpha, bins = bins)
    plt.xlabel(features)
    plt.title(title)
    plt.legend()

def plot(D0, D1, D2, propertyX, propertyY, featureX, featureY, title=""):
    plt.plot(D0[propertyX,:], D0[propertyY,:], 'o', label='Setosa', )
    plt.plot(D1[propertyX,:], D1[propertyY,:], 'o', label='Versicolor')
    plt.plot(D2[propertyX,:], D2[propertyY,:], 'o', label='Virginica')
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.title(title)
    plt.legend()


#LAB 03
def createGraphicPCA_LDA(L, dataProjectedPCA, dataProjectedLDA,):
    plt.figure("PCA vs LDA")
    plt.subplot(2, 2, 1)
    plot(dataProjectedPCA[:, L == 0], dataProjectedPCA[:, L == 1], dataProjectedPCA[:, L == 2], 0, 1, "", "", title="PCA, 1st and 2nd direction")
    
    plt.subplot(2, 2, 2)
    plot(dataProjectedLDA[:, L == 0], dataProjectedLDA[:, L == 1], dataProjectedLDA[:, L == 2], 0, 1, "", "", title="LDA, 1st and 2nd direction")

    plt.subplot(2, 2, 3)
    hist(dataProjectedPCA[:, L == 0], dataProjectedPCA[:, L == 1], dataProjectedPCA[:, L == 2], 0, "", title="PCA, 1st direction")

    plt.subplot(2, 2, 4)
    hist(dataProjectedLDA[:, L == 0], dataProjectedLDA[:, L == 1], dataProjectedLDA[:, L == 2], 0, "", title="LDA, 1st direction")
    plt.show()

def createGraphicTrainingLDA(projectedDataTraining, LTR, projectedDataValidation, LVAL):
    plt.figure("Training vs Validation")
    
    plt.subplot(1, 2, 1)
    hist(projectedDataTraining[:, LTR == 0], projectedDataTraining[:, LTR == 1], projectedDataTraining[:, LTR == 2], 0, "", title = "Training", bins = 5)

    plt.subplot(1, 2, 2)
    hist(projectedDataValidation[:, LVAL == 0], projectedDataValidation[:, LVAL == 1], projectedDataValidation[:, LVAL == 2], 0, "", title = "Validation", bins = 5)
    
    plt.show()