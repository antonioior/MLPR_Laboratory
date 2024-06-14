from SVM import SVM
from load import load_iris_binary
from utils import split_db_2to1

if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    SVM(DTR, LTR, DVAL, LVAL, printResult=True)
