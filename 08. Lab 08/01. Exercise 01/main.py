from function import numericalOptimization
from load import load_iris_binary
from logRegClass import binaryLogisticRegression
from utils import split_db_2to1

if __name__ == "__main__":
    printResult = True
    # Numerical Optimization
    numericalOptimization(printResult=printResult)

    # Binary Logistic regression
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    binaryLogisticRegression(DTR, LTR, DVAL, LVAL, printResult=printResult)
