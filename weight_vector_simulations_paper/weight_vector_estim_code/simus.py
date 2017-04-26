from datetime import datetime
import os
from multiprocessing import Pool
import pickle
import uuid
import numpy as np
from scipy.stats import multivariate_normal
from densities import generate_sample_f_star
from algorithm import WeightEstimator
from adaptiveDantzig import AdaptiveDantzigEstimator

FOLDER = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", ".") + "/"
print FOLDER
os.makedirs(FOLDER)
var = 10**(-4)
dist_rect = {
    (0,1./5) : 10./7,
    (1./5,2./5) : 5./7,
    (2./5,3./5) : 10./7,
    (3./5,4./5) : 0,
    (4./5,1) : 10./7
}
n_pdf = 100000

def simu(K, N, selected_densities):
    ###############
    #uniform case:#
    ###############
    print "start uniform"
    uniform_X, uniform_f_star = generate_sample_f_star(N, t="uniform", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect,  N_pdf=n_pdf)#
    #estim weight estimator
    cl = WeightEstimator(densities_dict=densities, select_threshold=10e-3)
    cl.fit(uniform_X)
    uniform_estim_weighted_densities=cl.select_densities()
    #dantzig estimator
    adapt_dantzig = AdaptiveDantzigEstimator(densities)
    uniform_lambda_dantzig = adapt_dantzig.fit(uniform_X)
    ###############
    #rect case:#
    ###############
    print "start rect"
    rect_X, rect_f_star = generate_sample_f_star(N, t="rect", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect, N_pdf=n_pdf)
    #estim weight estimator
    cl = WeightEstimator(densities_dict=densities, select_threshold=10e-3)
    cl.fit(rect_X)
    rect_estim_weighted_densities=cl.select_densities()
    #dantzig estimator
    adapt_dantzig = AdaptiveDantzigEstimator(densities)
    rect_lambda_dantzig = adapt_dantzig.fit(rect_X)
    ###############
    #cvx case    :#
    ###############
    print "start cvx"
    cvx_X, cvx_f_star, cvx_weights_star, _ = generate_sample_f_star(N, t="convex", densities=densities, cvx_rand=False, s=5, dist_rect=dist_rect,  N_pdf=n_pdf, selected_densities=selected_densities)
    #estim weight estimator
    cl = WeightEstimator(densities_dict=densities, select_threshold=10e-3)
    cl.fit(cvx_X)
    cvx_estim_weighted_densities=cl.select_densities()
    #dantzig estimator
    adapt_dantzig = AdaptiveDantzigEstimator(densities)
    cvx_lambda_dantzig = adapt_dantzig.fit(cvx_X)


    pickle.dump({"uniform_data": uniform_X,
                 "uniform_weight_vector_estim_lambda":uniform_estim_weighted_densities,
                 "uniform_adapative_dantzig":uniform_lambda_dantzig,
                 "uniform_f_star": uniform_f_star,
                 "rect_data": rect_X,
                 "rect_weight_vector_estim_lambda":rect_estim_weighted_densities,
                 "rect_adapative_dantzig":rect_lambda_dantzig,
                 "rect_f_star": rect_f_star,
                 "cvx_data": cvx_X,
                 "cvx_weights_star": cvx_weights_star,
                 "cvx_weight_vector_estim_lambda":cvx_estim_weighted_densities,
                 "cvx_adapative_dantzig":cvx_lambda_dantzig,
                 "cvx_f_star": cvx_f_star,
                 "densities" : densities,
                 "selected_densities" : selected_densities
             }, open(FOLDER +
                     "res_" + "K" + str(K) + "N" + str(N) + str(uuid.uuid4()), "wb"))

for K in [10, 50, 100]:
    if K == 10:
        selected_densities = [0, 2, 4, 6, 8]
    if K == 50:
        selected_densities = [0, 10, 20, 30, 40]
    if K == 100:
        selected_densities = [0, 20, 40, 60, 80]
    nodes = np.linspace(0, 1, K)
    densities = [multivariate_normal(m, var) for m in nodes]
    for N in [100, 500, 1000]:
        p = Pool(processes=9) 
        for _ in range(100):
            p.apply_async(simu, args=(K, N, selected_densities))
        p.close()
        p.join()