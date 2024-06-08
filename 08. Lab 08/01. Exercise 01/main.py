import numpy as np

from function import f, fWithGradient, computeMinFWithApproxGrad, computeMinFWithoutApproxGrad

if __name__ == "__main__":
    printResult = True
    computeMinFWithApproxGrad(inputFun=f, x0=np.array([0, 0]), iprint=-1, printResult=printResult)
    computeMinFWithoutApproxGrad(inputFun=fWithGradient, x0=np.array([0, 0]), iprint=-1, printResult=printResult)
