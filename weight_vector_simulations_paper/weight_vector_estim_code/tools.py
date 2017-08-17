# -*- coding: utf-8 -*-

from time import time
from itertools import combinations
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
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp


def get_results(folder, keys):
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
        
def l2_norm(f_over_g, f_sample, sample_size=100000, hypercube_size = 3):
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

def kl_norm(f_over_g, f_sample, sample_size=100000, hypercube_size = 3):
    """
    Compute the KL norm KL(f,g) using importance sampling
    with sample_size samples drawn from a gaussian mixture f_sample from pypmc
    input : 
    f_over_g : the integrand log(f/g) with f real density and g estimator of density,
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
    return 1./sample_size*np.apply_along_axis(f_over_g, 1, samples[0]).sum()

#def importance_sampling_integrate(f_over_g, f_sample, sample_size=10000, hypercube_size = 3):
#    """
#    Compute the L2 norm of f-g using importance sampling
#    with sample_size samples drawn from a gaussian mixture f_sample from pypmc
#    input : 
#    f_over_g : the integrand (1-g/f)^2*f with f real density and g estimator of density,
#               input is the pdf.
#    f_sample : sampling distrib f* known, type pypmc density mixture
#    """
#    # define indicator
#    dim = f_sample.dim
#    ind_lower = [-hypercube_size for _ in range(dim)]
#    ind_upper = [hypercube_size for _ in range(dim)]
#    ind = hyperrectangle(ind_lower, ind_upper)
#    sampler = ImportanceSampler(f_sample.evaluate, f_sample, ind)
#    sampler.run(sample_size)
#    samples = sampler.samples
#    return np.sqrt(1./sample_size*np.apply_along_axis(f_over_g, 1, samples[0]).sum())

class uniform_nonzero(object):
    """
    a wrapper to uniform density
    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def pdf(self, x):
        return uniform(self.loc, self.scale).pdf(x)+1e-20

def goodness_fit_densities(densities_list, SAMPLE_SIZE=1000):
    """
    'Multidimensional Goodness-of-fit from: On Multivariate Goodness–of–Fit and Two–Sample Testing'
    from Jerome H. Friedman. Performs a gof test on all combinations of 2 densities. 
    Transform the problem to a binary classification problem and a Kolmogorov-Smirnov test.
    input : a list of densities, some might be the same
    output: a new list of "unique" densities
    """
    new_dens = []
    for d1, d2 in list(combinations(densities_list, 2)):
        s = 0
        #This might not be statistically correct, but helps to stabiliize the result
        for _ in range(5):
            X1 = np.hstack([d1.rvs(2*SAMPLE_SIZE), np.ones(2*SAMPLE_SIZE).reshape(-1,1), np.arange(2*SAMPLE_SIZE).reshape(-1,1)])
            X2 = np.hstack([d2.rvs(2*SAMPLE_SIZE), -np.ones(2*SAMPLE_SIZE).reshape(-1,1), np.arange(2*SAMPLE_SIZE, 4*SAMPLE_SIZE).reshape(-1,1)])
            X = np.vstack([X1, X2])
            np.random.shuffle(X)
            indexes = X[:,-1]
            X = X[:,:-1]
            X_train = X[:SAMPLE_SIZE]
            X_test = X[SAMPLE_SIZE:2*SAMPLE_SIZE]
            indexes_test = X_test[:,-1]
            clf = LogisticRegression()
            #clf = RandomForestClassifier(max_depth=5, n_estimators=20)
            clf.fit(X_train[:,:-1], X_train[:,-1])
            scores = clf.predict_proba(X_test[:,:-1])
            s_plus = scores[indexes_test == 1][:,0]
            s_min = scores[indexes_test == -1][:,0]
            a = ks_2samp(s_plus, s_min)
            if a.pvalue > 0.3:
                s+=1
            print a.pvalue
        if s > 0.5 and d1 not in new_dens:
            new_dens.append(d1)
        else:
            if d1 not in new_dens:
                new_dens.append(d1)
            if d2 not in new_dens:
                new_dens.append(d2)
    return new_dens