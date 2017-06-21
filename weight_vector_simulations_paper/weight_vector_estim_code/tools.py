import numpy as np
from numba import jit
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from pypmc.sampler.importance_sampling import ImportanceSampler
from pypmc.tools.indicator import hyperrectangle

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

from pypmc.density.mixture import create_gaussian_mixture

def binary_centers(K,p):
    """
    Generate an array of K centers on the edges of the hypercube of dim p
    """
    if K > 2**p:
        print "Warning: not enough nodes"
        K_ = 2**p
    else:
        K_ = K
    centers = []
    for i in range(K_):
        bin_array = list(bin(i).split("b")[1])
        zeros_arr = [0]*(p-len(bin_array))
        centers.append(np.array(map(int, zeros_arr+bin_array)))
    return np.array(centers)

def generate_gaussian_mixture_sample(N, p, weights):
    """
    Generate a simple mixture with unit variances, the number of components are 
    given by the size of the vector of weights
    """
    K = len(weights)
    centers = binary_centers(K, p)
    cov = np.array([1e-2*np.diag(np.ones(p)) for _ in range(K)])
    mixture = create_gaussian_mixture(centers, cov, weights)
    return mixture.propose(N)
        

def l2_norm(f_over_g, f_sample, sample_size=10000, hypercube_size = 3):
    """
    Compute the L2 norm of f-g using importance sampling
    with sample_size samples drawn from a gaussian mixture f_sample from pypmc
    input : the integrand (1-g/f)^2*f, f sampling distrib f* known, g estimator of density.
    """
    # define indicator
    dim = f_sample.dim
    ind_lower = [-hypercube_size for _ in range(dim)]
    ind_upper = [hypercube_size for _ in range(dim)]
    ind = hyperrectangle(ind_lower, ind_upper)
    sampler = ImportanceSampler(f_sample.evaluate, f_sample, ind)
    sampler.run(sample_size)
    samples = sampler.samples
    return np.sqrt(1./sample_size*np.apply_along_axis(f_over_g, 1, samples[0]).sum())

class uniform_nonzero(object):
    """
    a wrapper to uniform density
    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def pdf(self, x):
        return uniform(self.loc, self.scale).pdf(x)+1e-20