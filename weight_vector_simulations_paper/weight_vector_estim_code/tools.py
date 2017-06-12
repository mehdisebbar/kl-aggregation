import numpy as np
from numba import jit
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform

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

def mle_bic(X, kmax):
    best_bic = 1e10
    best_model = None
    for k in range(2,kmax):
        cl = GaussianMixture(n_components=k)
        cl.fit(X)
        bic = cl.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_model = cl
    return best_bic, best_model

class uniform_nonzero(object):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def pdf(self, x):
        return uniform(self.loc, self.scale).pdf(x)+1e-20