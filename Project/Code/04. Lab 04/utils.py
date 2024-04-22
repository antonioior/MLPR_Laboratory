import numpy as np

# LAB 02
# Function to create a column vector 1


def mcol(data, shape):
    return data.reshape(shape, 1)


# Function to create a row vector
def mrow(data, shape):
    return data.reshape(1, shape)

# LAB 03


def projection(U, m):
    return U[:, ::-1][:, 0:m]


def split_db_2to1(D, L, seed=0):
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


def calculateError(DTR_lda, LTR, DVAL_lda, LVAL, printResults=False):
    # Original threshold
    # threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.
    # PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    # PVAL[DVAL_lda[0] >= threshold] = 1
    # PVAL[DVAL_lda[0] < threshold] = 0
    # difference = np.abs(LVAL - PVAL)
    # numOfErr = sum(x != 0 for x in difference)
    # errorRate = float(numOfErr) / float(LVAL.shape[0]) * 100

    # if printResults:
    #    print("Error - RESULTS")
    #    print(f"    Threshold: {threshold}")
    #    print(f"    Number of samples:\n\t{PVAL.shape[0]}")
    #    print(f"    Real values:\n\t{LVAL}")
    #    print(f"    Predicted values:\n\t{PVAL}")
    #    print(f"    Difference:\n\t{difference}")
    #    print(f"    Number of errors: {numOfErr}")
    #    print(f"    Error rate: {errorRate:.5f}%")

    # Compute of the best threshold
    best_threshold = None
    best_error_rate = float('inf')

    for threshold in np.linspace(DTR_lda.min(), DTR_lda.max(), 100):
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        diff = np.abs(PVAL - LVAL).sum()
        error_rate = diff / len(LVAL)
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_threshold = threshold

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= best_threshold] = 1
    PVAL[DVAL_lda[0] < best_threshold] = 0
    diff = np.abs(PVAL-LVAL).sum()
    error_rate = (diff / len(LVAL))

    if printResults:
        print("Error - RESULTS")
        print(f"    Threshold: {best_threshold}")
        print(f"    Number of samples:\n\t{PVAL.shape[0]}")
        print(f"    Real values:\n\t{LVAL}")
        print(f"    Predicted values:\n\t{PVAL}")
        print(f"    Number of errors: {diff}")
        print(f"    Error rate: {error_rate:.5f}%")


# LAB 04
def vcol(data):
    return data.reshape(data.shape[0], 1)


def vrow(data):
    return data.reshape(1, data.shape[0])


def logpdf_GAU_ND(X, mu, C):
    Y = []
    # Number of features
    N = X.shape[0]
    # Iter for each input data
    for x in X.T:
        x = vcol(x)
        const = N*np.log(2*np.pi)
        logC = np.linalg.slogdet(C)[1]
        mult = np.dot(np.dot((x-mu).T, np.linalg.inv(C)), (x-mu))[0, 0]
        Y.append(-0.5*(const + logC + mult))
    return np.array(Y)


def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()

def compute_mu_C(D):
    mu = mcol(D.mean(1), D.mean(1).size)
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C