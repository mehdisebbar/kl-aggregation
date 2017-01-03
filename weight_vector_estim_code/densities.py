import numpy as np
from scipy.stats import multivariate_normal
from numpy.random import multinomial
from random import sample

def generate_grid(grid_length, dim):
    # generate a grid of points in the cube [-1,1]^dim
    return np.vstack(np.meshgrid(*[np.linspace(0, 1, grid_length) for _ in range(dim)])).reshape(dim, -1).T

def check_grid_size(K, dim):
    # We make sure that K fills a cube, otherwise, we increase it
    # There are no reason to do so, purely esthetic
    if K**(1./dim)%1!=0:
        K = round(K ** (1. / dim))**dim
        print "K does not fill the cube, changing K=",K
    grid_length = K**(1. / dim)
    return int(K), int(grid_length)

def generate_densities(dim, var, grid_length):
    # We generate a dictionary of densities  at each point of the cube [-1,1]^dim
    # We want a cubic grid
    nodes = generate_grid(grid_length, dim)
    densities = [multivariate_normal(m, var * np.identity(dim)) for m in nodes]
    return densities

def generate_sample(N, K, dim, weights=None, n_densities=-1, var=10 ** (-2)):
    """
    :param K: Number of densities
    :param dim: Dimension
    :param weights: mixture weights
    :param n_densities: number of densities to select randomly
    :param s: Sparse index, default -1 : all densities are used,
              else a number of densities to be counted.
    :param N: Number of points
    :param var: variances for each components
    :return:
    """
    K, grid_length = check_grid_size(K, dim)
    if n_densities !=-1:
        selected_densities = sample(range(K), n_densities)
    else:
        selected_densities = range(K)
    # We check if the p-vals are given for the mixture
    if weights == None:
        weights = np.random.randint(N, size=(1, len(selected_densities)))[0]
        weights = 1.*weights/weights.sum()
    sample_repartition_among_clusters = multinomial(N, weights, size=1)[0]
    X = np.zeros([N,dim])
    densities_dict = generate_densities(dim, var, grid_length)
    t = 0
    for i in range(len(selected_densities)):
        X_d = densities_dict[selected_densities[i]].rvs(sample_repartition_among_clusters[i])
        X[t:t+X_d.shape[0]] = X_d
        t+=X_d.shape[0]
    np.random.shuffle(X)
    return X, densities_dict, selected_densities, weights