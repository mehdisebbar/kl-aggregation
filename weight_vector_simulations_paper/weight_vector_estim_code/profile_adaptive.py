from adaptiveDantzig import AdaptiveDantzigEstimator
from densities import generate_sample_f_star
from scipy.stats import multivariate_normal
import numpy as np

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
#X, f_star,  weights_star, selected_densities_star = generate_sample_f_star(1000, t="convex", densities=densities, cvx_rand=True, s=5, dist_rect=dist_rect,  N_pdf=n_pdf)
#X, f_star, weights_star, selected_densities_star = generate_sample_f_star(100, t="convex", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect,  N_pdf=n_pdf)
#plt.hist(X, bins=1000)  # plt.hist passes it's arguments to np.histogram
selected_densities = [2,5,10,15,30]

cvx_X, cvx_f_star, cvx_weights_star, _ = generate_sample_f_star(100, t="convex", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect,  N_pdf=n_pdf, selected_densities=selected_densities)
ad = AdaptiveDantzigEstimator(densities=densities)
res_ad = ad.fit(cvx_X)
print res_ad
