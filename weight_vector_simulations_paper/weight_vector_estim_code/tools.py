import numpy as np
from numba import jit

@jit()
def simplex_proj_numba(y):
    dim = len(y)
    u = np.flip(np.sort(y),0)
    maxi = 0
    lambd = 0
    for i in range(dim):
        crit = u[i]+1./(i+1)*(1-u[:i+1].sum())
        if crit > 0 and i > maxi:
            maxi = i
    s = u[:maxi+1].sum()
    lambd = 1./(maxi+1)*(1.-s)
    res = np.zeros(dim)
    for j in range(dim):
        res[j] = max(y[j]+lambd, 0)
    return res