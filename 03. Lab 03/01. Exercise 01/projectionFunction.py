#LAB 02
def mcol(mu, shape):
    return mu.reshape(shape, 1)

def mrow(mu, shape):
    return mu.reshape(1, shape)

#LAB 03
def projection(U, m):
    return U[:,::-1][:,0:m]