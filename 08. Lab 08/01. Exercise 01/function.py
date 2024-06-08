import numpy as np
import scipy.optimize as sp


def numericalOptimization(printResult=False):
    computeMinFWithApproxGrad(inputFun=f, x0=np.array([0, 0]), iprint=-1, printResult=printResult)
    computeMinFWithoutApproxGrad(inputFun=fWithGradient, x0=np.array([0, 0]), iprint=-1, printResult=printResult)


def f(x):
    y, z = x
    return (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2


def fWithGradient(x):
    y, z = x
    f = (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2
    gradientY = 2 * (y + 3) + np.cos(y)
    gradientZ = 2 * (z + 1)
    return f, np.array([gradientY, gradientZ])


def computeMinFWithApproxGrad(inputFun, x0, iprint, printResult=False):
    x, f, d = sp.fmin_l_bfgs_b(func=inputFun, x0=x0, approx_grad=True, iprint=iprint)

    if printResult:
        print("Result with approximated gradient:")
        print(f"\tMimimum is in {x}")
        print(f"\tValue to minimum point is {f: .9f}")
        print(f"\tNumer of time that f was called {d["funcalls"]}")


def computeMinFWithoutApproxGrad(inputFun, x0, iprint, printResult=False):
    x, f, d = sp.fmin_l_bfgs_b(func=inputFun, x0=x0, iprint=iprint)

    if printResult:
        print("Result without approximated gradient:")
        print(f"\tMimimum is in {x}")
        print(f"\tValue to minimum point is {f: .9f}")
        print(f"\tNumer of time that f was called {d["funcalls"]}")
