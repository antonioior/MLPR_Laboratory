from LinearSVM import linearSVM
from PolinomialSVM import polinomialSVM
from load import load_iris_binary
from utils import split_db_2to1

if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    linearSVM(DTR, LTR, DVAL, LVAL, printResult=True)
    polinomialSVM(DTR, LTR, DVAL, LVAL, printResult=True)
