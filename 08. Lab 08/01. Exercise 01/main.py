from function import numericalOptimization
from load import load_iris_binary
from logRegClass import binaryLogisticRegression
from priorWeightedLogClass import priorWeightedLogisticRegression
from utils import split_db_2to1

if __name__ == "__main__":
    printResult = True
    # Numerical Optimization
    numericalOptimization(printResult=printResult)

    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Binary Logistic regression
    binaryLogisticRegression(DTR, LTR, DVAL, LVAL, printResult=printResult)

    # PRIOR WEIGHTED LOGISTIC REGRESSION
    priorWeightedLogisticRegression(DTR, LTR, DVAL, LVAL, printResult=printResult)
