#LAB 2
#Function to load the data from the file
import numpy as np
import utils as ut

def load(fileName):
    data = open(fileName, 'r',)
    D = None
    L = np.array([], dtype=int)
    for row in data:
        values = row.strip().split(',')

        columnProperty = ut.mcol(np.array(values[0:6], dtype=float), 6)

        if D is None:
            D = columnProperty
        else:
            D = np.append(D, columnProperty, axis=1)

        L = np.append(L, int(values[6]))

    return D, L 