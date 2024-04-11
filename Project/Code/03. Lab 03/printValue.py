import projectionFunction

#Function to print the value of a matrix,
#formatted of one indentation

def printMatrix(matrix):
    print("\t[", end="")
    for i in range(0, matrix.shape[0]):
        if i == 0:
            print(f"{matrix[i,:]}")
        elif i == matrix.shape[0] - 1:
            print(f"\t {matrix[i,:]}", end="")
        else:
            print(f"\t {matrix[i,:]}")
    print("]", end="\n")


def printDataMain(muColumn, varColumn, C, var, std, D, printData = False):
    if printData:
        print("MAIN - RESULT")
        print("    Mean of the properties:")
        printMatrix(muColumn)
        print("    Variance of the properties:")
        printMatrix(varColumn)
        print("    Mean of the first two properties:")
        printMatrix(muColumn[0:2, :])
        print("    Variance of the first two properties:")
        printMatrix(varColumn[0:2, :])
        print("    Mean of the properties 3 and 4:")
        printMatrix(muColumn[2:4, :])
        print("    Variance of the properties 3 and 4:")
        printMatrix(varColumn[2:4, :])
        print("    Mean of the properties 5 and 6:")
        printMatrix(muColumn[4:6, :])
        print("    Variance of the properties 5 and 6:")
        printMatrix(varColumn[4:6, :])
        print("    Covariance matrix with dot product (DC.dot(DC.T)) / (D.shape[1])")
        printMatrix(C)
        print("    Variance is:")
        printMatrix(projectionFunction.mcol(var, D.shape[0]))
        print("    Std is:")
        printMatrix(projectionFunction.mcol(std, D.shape[0]))