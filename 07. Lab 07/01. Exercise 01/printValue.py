def printMatrix(matrix):
    print("\t[", end="")
    for i in range(0, matrix.shape[0]):
        if i == 0:
            print(f"{matrix[i, :]}")
        elif i == matrix.shape[0] - 1:
            print(f"\t {matrix[i, :]}", end="")
        else:
            print(f"\t {matrix[i, :]}")
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
