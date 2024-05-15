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