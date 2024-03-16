import numpy as np
import matplotlib.pyplot as plt

#Function to load the data from the file
def load(fileName):
    data = open(fileName, 'r',)
    D = None
    L = np.array([], dtype=int)
    for row in data:
        values = row.strip().split(',')

        columnProperty = mcol(np.array(values[0:6], dtype=float), 6)

        if D is None:
            D = columnProperty
        else:
            D = np.append(D, columnProperty, axis=1)

        L = np.append(L, int(values[6]))

    return D, L 

#Function to create a column vector
def mcol(data, shape):
    return data.reshape(shape, 1)

#Function to create a row vector
def mrow(data, shape):
    return data.reshape(1, shape)

#Functoin to create the graphics
def createGraphic(data, labels, properties, comment):
    data0 = data[:, labels == 0]
    data1 = data[:, labels == 1]
    for i in range(0, len(properties)):
        for j in range(0, len(properties)):
            if i == j:
                plt.figure(properties[i] + " " + comment)
                hist(data0, data1, i, properties[i])
            else:
                plt.figure(properties[i] + " " + properties[j] + " " + comment)
                scatter(data0, data1, i, j, properties[i], properties[j])
    plt.show()

#Function called by createGraphic to create the histogram
def hist(D0, D1, property, features):
    plt.hist(D0[property,:], density=True, label='Fake')
    plt.hist(D1[property,:], density=True, label='Real')
    plt.xlabel(features)
    plt.legend()

#Function called by createGraphic to create the scatter plot
def scatter(D0, D1, propertyX, propertyY, featureX, featureY):
    plt.scatter(D0[propertyX,:], D0[propertyY,:], label='Fake')
    plt.scatter(D1[propertyX,:], D1[propertyY,:], label='Real')
    plt.xlabel(featureX)
    plt.ylabel(featureY)
    plt.legend()


if __name__ == "__main__":
    D, L = load('trainData.txt')
    properties =["Features 1", "Features 2", "Features 3", "Features 4", "Features5" ,"Features 6"]
    
    #POINT 1, graphics only of the first two features
    createGraphic(D, L, properties[0:2], "")
    
    #POINT 2, graphics of second and third the features. The figure will be created only when you close the previous one
    createGraphic(D, L, properties[2:4], "")

    #POINT 3, graphics of the last two features. The figure will be created only when you close the previous one
    createGraphic(D, L, properties[4:6], "")