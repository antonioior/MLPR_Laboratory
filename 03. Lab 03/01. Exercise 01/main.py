from load import load
import graph 
import PCA
import LDA
import projectionFunction

if __name__ == '__main__':
    #LAB 02
    #VARIABLES TO PRINT
    createGraph = False
    printValue = False

    D, L = load('iris.csv')
    properties = ['Sepal Length', 'Sepal Width', 'Petal Lenght', 'Petal Width']

    if createGraph:
        graph.createGraphic(D, L, properties, "without mean")
    

    #STATISTICS
    #MEAN
    mu = D.mean(axis=1)
    muColumn = projectionFunction.mcol(mu, D.shape[0])
    muRow = projectionFunction.mrow(mu, D.shape[0])

    DC = D - muColumn

    #after closed all without mean it will print that with mean
    if createGraph:
        graph.createGraphic(DC, L, properties, "with mean")
    
    #COVARIANCE
    #print("Covariance matrix with np.cov(DC)\n", np.cov(DC))
    #C = (DC @ DC.T) / (D.shape[1])
    #print("Covariance matrix with formula (DC @ DC.T) / (D.shape[1])\n", C)
    C = (DC.dot(DC.T)) / (D.shape[1])
    var = D.var(1)  #variance is the square of std
    std = D.std(1)  
    
    
    if printValue:
        print("MAIN - RESULT")
        print(f"    Covariance matrix with dot product (DC.dot(DC.T)) / (D.shape[1])\n\t{C}")
        print(f"    Variance is:\n\t{projectionFunction.mcol(var, D.shape[0])}")
        print(f"    Std is:\n\t{projectionFunction.mcol(std, D.shape[0])}")

    #LAB 03
    #Calculate PCA
    dataProjectedPCA = PCA.PCA(D, L, C, printResults = False)

    #Calculate LDA
    dataProjectedLDA = LDA.LDA(D, L, printResults = True)

    #Print data
    graph.createGraphicPCA_LDA(L, dataProjectedPCA, dataProjectedLDA)

