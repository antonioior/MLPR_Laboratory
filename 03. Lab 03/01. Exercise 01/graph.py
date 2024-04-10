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


def hist(D0, D1, D2, property, features):
    plt.hist(D0[property,:], density=True, label='Setosa')
    plt.hist(D1[property,:], density=True, label='Versicolor')
    plt.hist(D2[property,:], density=True, label='Virginica')
    plt.xlabel(features)
    plt.legend()

def plot(D0, D1, D2, propertyX, propertyY, featureX, featureY):
    plt.plot(D0[propertyX,:], D0[propertyY,:], 'o', label='Setosa')
    plt.plot(D1[propertyX,:], D1[propertyY,:], 'o', label='Versicolor')
    plt.plot(D2[propertyX,:], D2[propertyY,:], 'o', label='Virginica')
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.legend()