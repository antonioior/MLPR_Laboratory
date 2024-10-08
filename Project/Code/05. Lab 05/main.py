import graph
import printValue
import utils as ut
from PCA_LDA import PCA_LDA
from densityEstimation import densityEstimation
from generativeModels import generativeModels
from load import load

if __name__ == "__main__":

    # VARIABLES TO PRINT
    createGraph = False

    # LAB 02 - COMPLETED
    D, L = load('trainData.txt')
    properties = ["Features 1", "Features 2", "Features 3", "Features 4", "Features 5", "Features 6"]

    mu = D.mean(axis=1)
    muColumn = ut.mcol(mu, D.shape[0])
    DC = D - muColumn

    var = D.var(1)
    std = D.std(1)  # square of variance
    varColumn = ut.mcol(var, D.shape[0])
    stdColumn = ut.mcol(std, D.shape[0])

    if createGraph:
        # POINT 1, graphics only of the first two features
        graph.createGraphic(D, L, properties, 0, 2, "without mean")
        graph.createGraphic(DC, L, properties, 0, 2, "with mean")
        graph.plt.show()

        # POINT 2, graphics of second and third the features.
        # The figure will be created only when you close the previous one
        graph.createGraphic(D, L, properties, 2, 4, "without mean")
        graph.createGraphic(DC, L, properties, 2, 4, "with mean")
        graph.plt.show()

        # POINT 3, graphics of the last two features. The figure will be created only when you close the previous one
        graph.createGraphic(D, L, properties, 4, 6, "without mean")
        graph.createGraphic(DC, L, properties, 4, 6, "with mean")
        graph.plt.show()

    # Covariance
    C = (DC.dot(DC.T)) / (D.shape[1])
    var = D.var(1)  # variance is the square of std
    std = D.std(1)
    printValue.printDataMain(muColumn, varColumn, C, var, std, D, printData=False)

    # LAB 03 - COMPLETED
    # PCA_LDA
    PCA_LDA(D, L, createGraph=False, printResults=False)

    # LAB 04 - COMPLETED
    # DENSITY ESTIMATION
    densityEstimation(D, L, createGraph=False, printResults=False, comment="")

    # LAB 05 - COMPLETED
    # GENERATIVE MODELS
    generativeModels(D, L, printResults=True)
