from algorithm import WeightEstimator
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal

n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
densities =[]
for x in np.linspace(-10,10,20):
    for y in np.linspace(-10,10,20):
        for var in [5, 1, 1e-1]:
            densities.append(multivariate_normal([x,y], var*np.diag([1,1])))

from algorithm import WeightEstimator
cl = WeightEstimator(densities_dict=densities, select_threshold=1e-2)
cl.fit(X)
res2 = cl.select_densities()

