from time import time
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from numba import jit
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from pypmc.sampler.importance_sampling import ImportanceSampler
from pypmc.tools.indicator import hyperrectangle
from pypmc.density.mixture import create_gaussian_mixture
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

def get_results(folder):
    """
    Get results from a folder at path, starting with res and produce a dataframe
    """
    onlyfiles = [f for f in listdir(folder) if (isfile(join(folder, f)) and f.startswith("res_K"))]
    res = []
    for f in onlyfiles:
        res.append(pickle.load(open(folder+f)))
    return pd.DataFrame(res)

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

class KdeCV(object):
    
    def __init__(self, n_jobs = 1, cv=5, bw = np.linspace(0.1, 1.0, 10)):
        self.grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': bw},
                                 cv=cv, n_jobs=n_jobs) # 20-fold cross-validation
    
    def fit(self, X):
        self.grid.fit(X)
        self.best_estimator = self.grid.best_estimator_

    def pdf(self, x):
        return np.exp(self.best_estimator.score_samples(x.reshape(1, -1)))

class GaussianMixtureGen(object):
    """
    Generate a simple mixture with unit variances, the number of components are 
    given by the size of the vector of weights
    """
    def __init__(self, p, weights):
        self.K = len(weights)
        self.weights = weights
        self.centers = self.binary_centers(self.K, p)
        self.cov = np.array([1e-2*np.diag(np.ones(p)) for _ in range(self.K)])
        
    def binary_centers(self, K,p):
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
        return np.array(centers, dtype=np.float)
    
    def get_params(self):
        return self.centers, self.cov
    
    def sample(self, N):
        mixture = create_gaussian_mixture(self.centers, self.cov, self.weights)
        return mixture.propose(N)
        
def l2_norm(f_over_g, f_sample, sample_size=10000, hypercube_size = 3):
    return importance_sampling_integrate(f_over_g, f_sample, sample_size=10000, hypercube_size = 3)

def kl_norm(f_over_g, f_sample, sample_size=10000, hypercube_size = 3):
    return importance_sampling_integrate(f_over_g, f_sample, sample_size=10000, hypercube_size = 3)

def importance_sampling_integrate(f_over_g, f_sample, sample_size=10000, hypercube_size = 3):
    """
    Compute the L2 norm of f-g using importance sampling
    with sample_size samples drawn from a gaussian mixture f_sample from pypmc
    input : 
    f_over_g : the integrand (1-g/f)^2*f with f real density and g estimator of density,
               input is the pdf.
    f_sample : sampling distrib f* known, type pypmc density mixture
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