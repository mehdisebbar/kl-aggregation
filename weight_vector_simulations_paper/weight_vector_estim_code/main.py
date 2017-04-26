
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from densities import generate_sample_f_star
from algorithm import WeightEstimator

nodes = np.linspace(0, 1, 50)
var = 10**(-4)
densities = [multivariate_normal(m, var) for m in nodes]
dist_rect = {
    (0,1./5) : 10./7,
    (1./5,2./5) : 5./7,
    (2./5,3./5) : 10./7,
    (3./5,4./5) : 0,
    (4./5,1) : 10./7
}
n_pdf = 100000
#X, f_star = generate_sample_f_star(10000, t="rect", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect, N_pdf=n_pdf)
#X, f_star = generate_sample_f_star(1000, t="uniform", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect,  N_pdf=n_pdf)#
X, f_star,  weights_star, selected_densities_star = generate_sample_f_star(1000, t="convex", densities=densities, cvx_rand=True, s=5, dist_rect=dist_rect,  N_pdf=n_pdf)
#X, f_star, weights_star, selected_densities_star = generate_sample_f_star(100, t="convex", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect,  N_pdf=n_pdf)
#plt.hist(X, bins=1000)  # plt.hist passes it's arguments to np.histogram

#estimation
cl = WeightEstimator(densities_dict=densities, select_threshold=10e-3)
cl.fit(X)
estim_weighted_densities=cl.select_densities()
selected_densities_estim, weights_estim = np.array(zip(*estim_weighted_densities))
f_estim = np.apply_along_axis(lambda x: weights_estim.dot(np.array([densities[i].pdf(x) for i in selected_densities_estim.astype(int)])), 0, np.linspace(0,1,n_pdf))

plt.plot(np.linspace(0,1,n_pdf),f_star)
plt.plot(np.linspace(0,1,n_pdf),f_estim)
plt.show()

#Kullback Leibler:

from scipy.stats import entropy
#import scipy.integrate as integrate
#integrate.quad(lambda x: x**2, 0, 1)

print "KL-div:", entropy(f_star, f_estim)
print "L2 norm", 1./n_pdf*np.linalg.norm(f_star-f_estim,axis=0)**2
print weights_estim, weights_star