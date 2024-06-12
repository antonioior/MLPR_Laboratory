import utils as ut


# Function to print the value of a matrix,
# formatted of one indentation


def printMatrix(matrix, decimals=8):
    print("\t[", end="")
    for i in range(0, matrix.shape[0]):
        formatted_row = ", ".join(f"{num:.{decimals}f}" for num in matrix[i, :])
        if i == 0:
            print(f"[{formatted_row}]")
        elif i == matrix.shape[0] - 1:
            print(f"\t [{formatted_row}]", end="")
        else:
            print(f"\t [{formatted_row}]")
    print("]", end="\n")


def printConfusionMatrix(matrix):
    spaces = 14
    print(f"\t\t{spaces * " " + 2 * " "}Class:")
    for i in range(matrix.shape[0]):
        print("\t\t", end="")
        if i == 0:
            print(f"{spaces * " " + 2 * " "}|  {' '.join(str(k) for k in range(matrix.shape[1]))}")
            print(f"\t\t{spaces * " "}-----------")
        elif i == 1:
            print(f"Prediction", end="")
        for j in range(matrix.shape[1]):
            if j == 0 and i == 0:
                print(f"\t\t{spaces * " "}{i} | ", end="")
            elif j == 0 and i != 1:
                print(f"{spaces * " "}{i} | ", end="")
            elif j == 0 and i == 1:
                print(f"{4 * " "}{i} | ", end="")
            print(f"{matrix[i][j]}", end=" ")
        print()
    print()


def printDataMain(muColumn, varColumn, C, var, std, D, printData=False):
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
        printMatrix(ut.mcol(var, D.shape[0]))
        print("    Std is:")
        printMatrix(ut.mcol(std, D.shape[0]))
