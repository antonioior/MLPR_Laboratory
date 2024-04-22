import numpy as np

#LAB 02
#Function to create a column vector 1
def mcol(data, shape):
    return data.reshape(shape, 1)

#Function to create a row vector
def mrow(data, shape):
    return data.reshape(1, shape)

#LAB 03
def projection(U, m):
    return U[:,::-1][:,0:m]

def split_db_2to1(D, L, seed = 0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

def calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults = False):
    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    difference = np.abs(LVAL - PVAL)
    numOfErr = sum( x != 0 for x in difference)
    errorRate = float(numOfErr) / float(LVAL.shape[0]) * 100 

    if printResults:
        print("Error - RESULTS")
        print(f"    Threshold: {threshold}")
        print(f"    Number of samples:\n\t{PVAL.shape[0]}")
        print(f"    Real values:\n\t{LVAL}")
        print(f"    Predicted values:\n\t{PVAL}")
        print(f"    Difference:\n\t{difference}")
        print(f"    Number of errors: {numOfErr}")
        print(f"    Error rate: {errorRate:.5f}%")