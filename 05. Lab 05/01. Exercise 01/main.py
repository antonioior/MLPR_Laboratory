from MVG import MVG
from load import load
from utils import split_db_2to1

if __name__ == "__main__":
    printData = False
    D, L = load()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # First model Multivariate Gaussian Classifier
    MVG(DTR, LTR, DTE, LTE, printData=False)
