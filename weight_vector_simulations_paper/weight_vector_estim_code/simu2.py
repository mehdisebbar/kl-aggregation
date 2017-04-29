from datetime import datetime
import os
from multiprocessing import Pool
import pickle
import uuid
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import laplace
from scipy.integrate import simps
from DensitiesGenerator import DensityGenerator
from scipy.stats import gaussian_kde
from pythonABC.hselect import hsj
from algorithm import WeightEstimator
from adaptiveDantzig import AdaptiveDantzigEstimator

N_PDF = 10000
SELECT_THRESHOLD = 10e-5
FOLDER = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", ".") + "/"
os.makedirs(FOLDER)

dist_rect = {
    (0,1./5) : 10./7,
    (1./5,2./5) : 5./7,
    (2./5,3./5) : 10./7,
    (3./5,4./5) : 0,
    (4./5,1) : 10./7
}

def gaussian_densities_generator(K, var=10**(-4)):
    #Generate a list of K gaussians reparted in [0,1]
    nodes = np.linspace(0, 1, K)
    return [multivariate_normal(m, var) for m in nodes]

def laplacian_densities_generator(K, scale=0.2):
    #Generate a list of K Laplacians reparted in [0,1]
    nodes = np.linspace(0, 1, K)
    return [laplace(m, scale) for m in nodes]

def normalize_density(f_pdf):
    #return a normalized vector such that the integral is 1
    return f_pdf/simps(f_pdf, np.linspace(0, 1, N_PDF))

def simu_block(X, densities, cl, adapt_dantzig):
    try:
        print "MLE",
        cl.fit(X)
        estim_weighted_densities=cl.select_densities()
        #dantzig estimator
        print "AD",
        lambda_dantzig = adapt_dantzig.fit(X)
    except:
        print "Error: Cannot compute"
        raise 
    #kde with Sheater-Jones bandwith selection method
    print "KDE-SJ",
    kernel = gaussian_kde(X, bw_method=hsj(X))
    pdf_kde_hsj = kernel.pdf(np.linspace(0,1,N_PDF))
    print "KDE"
    kernel = gaussian_kde(X)
    pdf_kde = kernel.pdf(np.linspace(0,1,N_PDF))
    return estim_weighted_densities, lambda_dantzig, pdf_kde_hsj, pdf_kde

def simu(K, N):
    dg = DensityGenerator(n_pdf= N_PDF)
    cl = WeightEstimator(densities_dict=densities, select_threshold=SELECT_THRESHOLD)
    adapt_dantzig = AdaptiveDantzigEstimator(densities)

    ###############
    #uniform case:#
    ###############
    print "uniform",
    uniform_X, uniform_f_star = dg.generate_uniform(n_points=N)
    uniform_f_star = normalize_density(uniform_f_star)
    try:
        uniform_estim_weighted_densities, uniform_lambda_dantzig, uniform_kde_pdf_hsj, uniform_kde_pdf = simu_block(uniform_X, densities, cl, adapt_dantzig)   
    except:
        return 0

    ###############
    #rect case:#
    ###############
    print "rect",
    rect_X, rect_f_star = dg.generate_rect(N, dist_rect)
    rect_f_star = normalize_density(rect_f_star)
    try:
        rect_estim_weighted_densities, rect_lambda_dantzig, rect_kde_pdf_hsj, rect_kde_pdf = simu_block(rect_X, densities, cl, adapt_dantzig)
    except:
        return 0

    ###############
    #f* 5 gaussians, same weights
    # mean k/5, var=10^(-4)
    ###############
    print "5 gaussians, same weights",
    var = 10**(-3)
    selected_densities_gauss = []
    for m in [0.2, 0.4, 0.6, 0.8, 1]:
        selected_densities_gauss.append(multivariate_normal(m, var))
    gauss_X, gauss_f_star, gauss_weights_star, _ = dg.gaussian(n_points=N, densities=selected_densities_gauss, selected_densities=range(5))
    gauss_f_star = normalize_density(gauss_f_star)
    try:
        gauss_estim_weighted_densities, gauss_lambda_dantzig, gauss_kde_pdf_hsj, gauss_kde_pdf = simu_block(gauss_X, densities, cl, adapt_dantzig)
    except:
        return 0

    ###############
    #f* mix gaussian/laplace in dict
    ###############
    print "f* mix gaussian/laplace in dict",
    selected_densities_lapl_gauss = []
    selected_densities_lapl_gauss.append(multivariate_normal(0.2, 10**(-3)))
    selected_densities_lapl_gauss.append(multivariate_normal(0.6, 10**(-3)))
    selected_densities_lapl_gauss.append(multivariate_normal(0, 10**(-2)))
    selected_densities_lapl_gauss.append(laplace(0.4,0.2))
    selected_densities_lapl_gauss.append(laplace(0.8,0.1))
    lapl_gauss_X, lapl_gauss_f_star, lapl_gauss_weights_star, _ = dg.gaussian(n_points=N, densities=selected_densities_lapl_gauss, selected_densities=range(5))
    lapl_gauss_f_star = normalize_density(lapl_gauss_f_star)
    try:
        lapl_gauss_estim_weighted_densities, lapl_gauss_lambda_dantzig, lapl_gauss_kde_pdf_hsj, lapl_gauss_kde_pdf = simu_block(lapl_gauss_X, densities, cl, adapt_dantzig)
    except:
        return 0
    
    ###############
    #f* Another with densities not in dict
    ###############
    print "f* with densities not in dict",
    selected_densities_lapl_gauss_not_dict = []
    selected_densities_lapl_gauss_not_dict.append(multivariate_normal(0.1, 5*10**(-3)))
    selected_densities_lapl_gauss_not_dict.append(multivariate_normal(0.6, 10**(-3)))
    selected_densities_lapl_gauss_not_dict.append(multivariate_normal(0, 10**(-2)))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.4, 0.1))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.8, 0.1))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.3, 0.1))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.7, 0.1))
    lapl_gauss_not_dict_X, lapl_gauss_not_dict_f_star, lapl_gauss_not_dict_weights_star, _ = dg.gaussian(n_points=N, densities=selected_densities_lapl_gauss_not_dict, selected_densities=range(7))
    lapl_gauss_not_dict_f_star = normalize_density(lapl_gauss_not_dict_f_star)
    try:
        lapl_gauss_not_dict_estim_weighted_densities, lapl_gauss_not_dict_lambda_dantzig, lapl_gauss_not_dict_kde_pdf_hsj, lapl_gauss_not_dict_kde_pdf = simu_block(lapl_gauss_not_dict_X, densities, cl, adapt_dantzig)
    except:
        return 0
    
    print "OK, writing results"
    pickle.dump({"uniform_data": uniform_X,
                 "uniform_weight_vector_estim_lambda":uniform_estim_weighted_densities,
                 "uniform_adapative_dantzig":uniform_lambda_dantzig,
                 "uniform_kde_hsj": uniform_kde_pdf_hsj,
                 "uniform_kde": uniform_kde_pdf,
                 "uniform_f_star": uniform_f_star,

                 "rect_data": rect_X,
                 "rect_weight_vector_estim_lambda":rect_estim_weighted_densities,
                 "rect_adapative_dantzig":rect_lambda_dantzig,
                 "rect_kde_hsj": rect_kde_pdf_hsj,
                 "rect_kde": rect_kde_pdf,
                 "rect_f_star": rect_f_star,

                 "gauss_data": gauss_X,
                 "gauss_weights_star": gauss_weights_star,
                 "gauss_weight_vector_estim_lambda": gauss_estim_weighted_densities,
                 "gauss_adapative_dantzig": gauss_lambda_dantzig,
                 "gauss_f_star": gauss_f_star,
                 "gauss_kde_hsj": gauss_kde_pdf_hsj,
                 "gauss_kde": gauss_kde_pdf,
                 "gauss_selected_densities" : selected_densities_gauss,

                 "lapl_gauss_data": lapl_gauss_X,
                 "lapl_gauss_weights_star": lapl_gauss_weights_star,
                 "lapl_gauss_weight_vector_estim_lambda":lapl_gauss_estim_weighted_densities,
                 "lapl_gauss_adapative_dantzig":lapl_gauss_lambda_dantzig,
                 "lapl_gauss_f_star": lapl_gauss_f_star,
                 "lapl_gauss_kde_hsj": lapl_gauss_kde_pdf_hsj,
                 "lapl_gauss_kde": lapl_gauss_kde_pdf,
                 "lapl_gauss_selected_densities" : selected_densities_lapl_gauss,

                 "lapl_gauss_not_dict_data": lapl_gauss_not_dict_X,
                 "lapl_gauss_not_dict_weights_star": lapl_gauss_not_dict_weights_star,
                 "lapl_gauss_not_dict_weight_vector_estim_lambda": lapl_gauss_not_dict_estim_weighted_densities,
                 "lapl_gauss_not_dict_adapative_dantzig":lapl_gauss_not_dict_lambda_dantzig,
                 "lapl_gauss_not_dict_f_star": lapl_gauss_not_dict_f_star,
                 "lapl_gauss_not_dict_kde_hsj": lapl_gauss_not_dict_kde_pdf_hsj,
                 "lapl_gauss_not_dict_kde": lapl_gauss_not_dict_kde_pdf,
                 "lapl_gauss_not_dict_selected_densities" : selected_densities_lapl_gauss_not_dict,
                 
                 "densities" : densities,
                 "N" : N
             }, open(FOLDER +
                     "res_" + "K" + str(K) + "N" + str(N) +"_"+ str(uuid.uuid4()), "wb"))
    return 1


if __name__ == "__main__":
    print FOLDER
    print "Generating the dictionary"
    nodes_gauss = [0, 0.2, 0.4, 0.6, 0.8, 1]
    var_list = [1, 10**(-1), 10**(-2), 10**(-3)]
    densities = []
    for m in nodes_gauss:
        for var in var_list:
            densities.append(multivariate_normal(m, var))
    scales =[0.05, 0.1, 0.5, 1]
    nodes_lapl = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for m in nodes_lapl:
        for scale in scales:
            densities.append(laplace(loc=m, scale=scale))

    for N in [100, 500, 1000]:
        p = Pool(processes=9) 
        i=0
        #We send a batch of 20 tasks
        while i <=200:
            res = [p.apply_async(simu, args=(0, N)) for _ in range(20)]
            i += sum([r.get() for r in res])
        p.close()
        p.join()        