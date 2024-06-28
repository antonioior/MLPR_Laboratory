import numpy as np
import scipy as sp


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
    diff = np.abs(PVAL - LVAL).sum()
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
        const = N * np.log(2 * np.pi)
        logC = np.linalg.slogdet(C)[1]
        mult = np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x - mu))[0, 0]
        Y.append(-0.5 * (const + logC + mult))
    return np.array(Y)


def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()


def compute_mu_C(D):
    mu = mcol(D.mean(1), D.mean(1).size)
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


# LAB 05


def logpdf_GAU_ND_prof(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (
            (x - mu) * (P @ (x - mu))).sum(0)


def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = vrow(sp.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost


def compute_covariance_I_Class(D):
    mu = D.mean(axis=1)
    mu = vcol(mu)
    dataCenter = D - mu
    covariance = (dataCenter @ dataCenter.T) / float(dataCenter.shape[1])
    return covariance


def correlationMatrix(C):
    Corr = C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))
    return Corr


# LAB 06 - NO PROJECT PART

# LAB 07
def optimalBayesDecision(pi, Cfn, Cfp):
    threshold = - np.log(pi * Cfn / ((1 - pi) * Cfp))
    return threshold


def computePVAL(llr, pi, Cfn, Cfp):
    t = optimalBayesDecision(pi, Cfn, Cfp)
    PVAL = np.zeros(len(llr), dtype=int)
    for i in range(len(llr)):
        PVAL[i] = 1 if llr[i] > t else 0
    return PVAL


def computeConfusionMatrix(PVAL, LTE):
    num_classes = len(set(LTE))
    confusionMatrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(PVAL)):
        confusionMatrix[PVAL[i]][LTE[i]] += 1
    return confusionMatrix


# LAB 08
def errorRate(DVAL, LVAL, vf, K=1):
    w, b = vf[:-1], vf[-1] * K
    sVal = np.dot(w.T, DVAL).ravel() + b
    Pval = (sVal > 0) * 1
    error = (Pval != LVAL).mean()
    return error, sVal


def z_normalizeTraining(D):
    return (D - D.mean(axis=1).reshape(-1, 1)) / D.std(axis=1).reshape(-1, 1)


def z_normalizeOther(DT, D):
    return (D - DT.mean(axis=1).reshape(-1, 1)) / DT.std(axis=1).reshape(-1, 1)


# LAB 10
def logpdf_GMM(X, gmm):
    M = len(gmm)
    N = X.shape[1]
    S = np.zeros((M, N))
    for g in range(M):
        w, mu, C = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    logdens = sp.special.logsumexp(S, axis=0)
    return S, logdens
