import sklearn.datasets


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L
