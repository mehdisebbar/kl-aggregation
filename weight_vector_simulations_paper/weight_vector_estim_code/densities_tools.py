import numpy as np
from scipy.stats import multivariate_normal
from numpy.random import multinomial
from random import sample

def generate_grid(grid_length, dim):
    # generate grid_length centers in [0,1]
    return np.vstack(np.meshgrid(*[np.linspace(0, 1, grid_length) for _ in range(dim)])).reshape(dim, -1).T
