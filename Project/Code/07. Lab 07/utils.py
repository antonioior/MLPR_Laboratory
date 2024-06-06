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


def computeDCF(confusionMatrix, pi, Cfn, Cfp, normalized=True):
    Pfn = confusionMatrix[0][1] / (confusionMatrix[1][1] + confusionMatrix[0][1])
    Pfp = confusionMatrix[1][0] / (confusionMatrix[0][0] + confusionMatrix[1][0])
    dcfValue = pi * Cfn * Pfn + (1 - pi) * Cfp * Pfp
    if normalized:
        Bdummy = min(pi * Cfn, (1 - pi) * Cfp)
        return dcfValue / Bdummy
    return dcfValue


def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]  # We sort the llrs
    classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []

    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse

    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)

    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
            idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])

    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)  # we return also the corresponding thresholds


# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (
            1 - prior) * Cfp)  # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


def computeEffectivePrior(prior):
    return 1 / (1 + np.exp(-prior))


def bayesError(llr, LTE, lineLeft, lineRight):
    effPriorLogOdds = np.linspace(lineLeft, lineRight, 21)
    piT = []
    dcf = []
    minDCF = []
    for prior in effPriorLogOdds:
        piT.append(computeEffectivePrior(prior))

    for effectivePrior in piT:
        PVAL = computePVAL(llr, effectivePrior, 1, 1)
        matrix = computeConfusionMatrix(PVAL, LTE)
        dcf.append(computeDCF(matrix, effectivePrior, 1, 1, True))
        minDCF.append(compute_minDCF_binary_fast(llr, LTE, effectivePrior, 1, 1))
    return effPriorLogOdds, dcf, minDCF
