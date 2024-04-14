import numpy as np

#LAB 02
def mcol(mu, shape):
    return mu.reshape(shape, 1)

def mrow(mu, shape):
    return mu.reshape(1, shape)

#LAB 03
def projection(U, m):
    return U[:,::-1][:,0:m]

#LAB 03
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