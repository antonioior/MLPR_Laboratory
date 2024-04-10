from load import load
import graph 
import PCA
import LDA
import projectionFunction

if __name__ == '__main__':
    #LAB 02
    createGraph = False
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
    print("Covariance matrix with dot product (DC.dot(DC.T)) / (D.shape[1])\n", C)
    var = D.var(1)  #variance is the square of std
    std = D.std(1)  
    print("Variance is:\n", projectionFunction.mcol(var, D.shape[0]))
    print("Std is:\n", projectionFunction.mcol(std, D.shape[0]))
    
    #LAB 03
    #Calculate PCA
    PCA.PCA(D, L, C)

    #Calculate LDA
    LDA.LDA(D, L)
