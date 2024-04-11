#LAB 02

#Function to create a column vector
def mcol(data, shape):
    return data.reshape(shape, 1)

#Function to create a row vector
def mrow(data, shape):
    return data.reshape(1, shape)

#LAB 03
def projection(U, m):
    return U[:,::-1][:,0:m]