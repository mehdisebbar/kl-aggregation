from datetime import datetime
import os
from multiprocessing import Pool
import pickle
import uuid
import numpy as np
from scipy.stats import multivariate_normal, laplace
from scipy.integrate import simps
from DensitiesGenerator import DensityGenerator
from scipy.stats import gaussian_kde
from pythonABC.hselect import hsj
from algorithm import WeightEstimator
from adaptiveDantzig import AdaptiveDantzigEstimator
from tools import mle_bic, uniform_nonzero
from time import time

N_PDF = 10000
SELECT_THRESHOLD = 10e-5
MAX_COMPONENTS_MLE_BIC = 20
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
        a=time()
        cl.fit(X)
        estim_weighted_densities=cl.pi_final
        b=time()
        time_mle = b-a
        #dantzig estimator
        print "AD",
        a=time()
        lambda_dantzig = adapt_dantzig.fit(X)
        b=time()
        time_ad = b-a
    except:
        print "Error: Cannot compute"
        raise 
    #kde with Sheater-Jones bandwith selection method
    print "KDE-SJ",
    a=time()
    kernel = gaussian_kde(X, bw_method=hsj(X))
    b=time()
    time_kde_sj = b-a
    pdf_kde_hsj = kernel.pdf(np.linspace(0,1,N_PDF))
    print "KDE",
    a=time()
    kernel = gaussian_kde(X)
    b=time()
    time_kde = b-a
    pdf_kde = kernel.pdf(np.linspace(0,1,N_PDF))
    print "MLE+bic"
    Y = X.reshape(-1,1)
    a=time()
    mle_bic_val, mle_bic_model = mle_bic(Y, MAX_COMPONENTS_MLE_BIC)
    b=time()
    time_mle_bic = b-a
    print "Done"
    return estim_weighted_densities, lambda_dantzig, pdf_kde_hsj, pdf_kde, mle_bic_val, mle_bic_model, time_ad, time_kde, time_kde_sj, time_mle, time_mle_bic

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
        uniform_estim_weighted_densities, uniform_lambda_dantzig, uniform_kde_pdf_hsj, uniform_kde_pdf, uniform_mle_bic, uniform_mle_bic_model, uniform_time_ad, uniform_time_kde, uniform_time_kde_sj, uniform_time_mle, uniform_time_mle_bic = simu_block(uniform_X, densities, cl, adapt_dantzig)   
    except Exception as e:
        print e
        return 0

    ###############
    #rect case:#
    ###############
    print "rect",
    rect_X, rect_f_star = dg.generate_rect(N, dist_rect)
    rect_f_star = normalize_density(rect_f_star)
    try:
        rect_estim_weighted_densities, rect_lambda_dantzig, rect_kde_pdf_hsj, rect_kde_pdf, rect_mle_bic, rect_mle_bic_model, rect_time_ad, rect_time_kde, rect_time_kde_sj, rect_time_mle, rect_time_mle_bic = simu_block(rect_X, densities, cl, adapt_dantzig)
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
        gauss_estim_weighted_densities, gauss_lambda_dantzig, gauss_kde_pdf_hsj, gauss_kde_pdf,gauss_mle_bic, gauss_mle_bic_model, gauss_time_ad, gauss_time_kde, gauss_time_kde_sj, gauss_time_mle, gauss_time_mle_bic = simu_block(gauss_X, densities, cl, adapt_dantzig)
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
        lapl_gauss_estim_weighted_densities, lapl_gauss_lambda_dantzig, lapl_gauss_kde_pdf_hsj, lapl_gauss_kde_pdf, lapl_gauss_mle_bic, lapl_gauss_mle_bic_model, lapl_gauss_time_ad, lapl_gauss_time_kde, lapl_gauss_time_kde_sj, lapl_gauss_time_mle, lapl_gauss_time_mle_bic= simu_block(lapl_gauss_X, densities, cl, adapt_dantzig)
    except:
        return 0
    
    ###############
    #f* Another with densities not in dict
    ###############
    print "f* with densities not in dict",
    selected_densities_lapl_gauss_not_dict = []
    selected_densities_lapl_gauss_not_dict.append(multivariate_normal(0.1, 5*10**(-3)))
    selected_densities_lapl_gauss_not_dict.append(multivariate_normal(0.65, 10**(-3)))
    selected_densities_lapl_gauss_not_dict.append(multivariate_normal(0.9, 10**(-2)))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.5, 0.08))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.2, 0.07))
    selected_densities_lapl_gauss_not_dict.append(laplace(0.75, 0.05))
    lapl_gauss_not_dict_X, lapl_gauss_not_dict_f_star, lapl_gauss_not_dict_weights_star, _ = dg.gaussian(n_points=N, densities=selected_densities_lapl_gauss_not_dict, selected_densities=range(len(selected_densities_lapl_gauss_not_dict)))
    lapl_gauss_not_dict_f_star = normalize_density(lapl_gauss_not_dict_f_star)
    try:
        lapl_gauss_not_dict_estim_weighted_densities, lapl_gauss_not_dict_lambda_dantzig, lapl_gauss_not_dict_kde_pdf_hsj, lapl_gauss_not_dict_kde_pdf,  lapl_gauss_not_dict_mle_bic, lapl_gauss_not_dict_mle_bic_model, lapl_gauss_not_dict_time_ad, lapl_gauss_not_dict_time_kde, lapl_gauss_not_dict_time_kde_sj, lapl_gauss_not_dict_time_mle, lapl_gauss_not_dict_time_mle_bic= simu_block(lapl_gauss_not_dict_X, densities, cl, adapt_dantzig)
    except:
        return 0
    
    print "OK, writing results"
    pickle.dump({"uniform_data": uniform_X,
                 "uniform_weight_vector_estim_lambda":uniform_estim_weighted_densities,
                 "uniform_adapative_dantzig":uniform_lambda_dantzig,
                 "uniform_kde_hsj": uniform_kde_pdf_hsj,
                 "uniform_kde": uniform_kde_pdf,
                 "uniform_f_star": uniform_f_star,
                 "uniform_mle_bic_model": uniform_mle_bic_model,
                 "uniform_mle_time": uniform_time_mle,
                 "uniform_mle_bic_time": uniform_time_mle_bic,
                 "uniform_kde_time": uniform_time_kde,
                 "uniform_kde_sj_time": uniform_time_kde_sj,
                 "uniform_ad_time": uniform_time_ad,

                 "rect_data": rect_X,
                 "rect_weight_vector_estim_lambda":rect_estim_weighted_densities,
                 "rect_adapative_dantzig":rect_lambda_dantzig,
                 "rect_kde_hsj": rect_kde_pdf_hsj,
                 "rect_kde": rect_kde_pdf,
                 "rect_f_star": rect_f_star,
                 "rect_mle_bic_model": rect_mle_bic_model,
                 "rect_mle_time": rect_time_mle,
                 "rect_mle_bic_time": rect_time_mle_bic,
                 "rect_kde_time": rect_time_kde,
                 "rect_kde_sj_time": rect_time_kde_sj,
                 "rect_ad_time": rect_time_ad,

                 "gauss_data": gauss_X,
                 "gauss_weights_star": gauss_weights_star,
                 "gauss_weight_vector_estim_lambda": gauss_estim_weighted_densities,
                 "gauss_adapative_dantzig": gauss_lambda_dantzig,
                 "gauss_f_star": gauss_f_star,
                 "gauss_kde_hsj": gauss_kde_pdf_hsj,
                 "gauss_kde": gauss_kde_pdf,
                 "gauss_selected_densities" : selected_densities_gauss,
                 "gauss_mle_bic_model": gauss_mle_bic_model,
                 "gauss_mle_time": gauss_time_mle,
                 "gauss_mle_bic_time": gauss_time_mle_bic,
                 "gauss_kde_time": gauss_time_kde,
                 "gauss_kde_sj_time": gauss_time_kde_sj,
                 "gauss_ad_time": gauss_time_ad,

                 "lapl_gauss_data": lapl_gauss_X,
                 "lapl_gauss_weights_star": lapl_gauss_weights_star,
                 "lapl_gauss_weight_vector_estim_lambda":lapl_gauss_estim_weighted_densities,
                 "lapl_gauss_adapative_dantzig":lapl_gauss_lambda_dantzig,
                 "lapl_gauss_f_star": lapl_gauss_f_star,
                 "lapl_gauss_kde_hsj": lapl_gauss_kde_pdf_hsj,
                 "lapl_gauss_kde": lapl_gauss_kde_pdf,
                 "lapl_gauss_selected_densities" : selected_densities_lapl_gauss,
                 "lapl_gauss_mle_bic_model": lapl_gauss_mle_bic_model,
                 "lapl_gauss_mle_time": lapl_gauss_time_mle,
                 "lapl_gauss_mle_bic_time": lapl_gauss_time_mle_bic,
                 "lapl_gauss_kde_time": lapl_gauss_time_kde,
                 "lapl_gauss_kde_sj_time": lapl_gauss_time_kde_sj,
                 "lapl_gauss_ad_time": lapl_gauss_time_ad,

                 "lapl_gauss_not_dict_data": lapl_gauss_not_dict_X,
                 "lapl_gauss_not_dict_weights_star": lapl_gauss_not_dict_weights_star,
                 "lapl_gauss_not_dict_weight_vector_estim_lambda": lapl_gauss_not_dict_estim_weighted_densities,
                 "lapl_gauss_not_dict_adapative_dantzig":lapl_gauss_not_dict_lambda_dantzig,
                 "lapl_gauss_not_dict_f_star": lapl_gauss_not_dict_f_star,
                 "lapl_gauss_not_dict_kde_hsj": lapl_gauss_not_dict_kde_pdf_hsj,
                 "lapl_gauss_not_dict_kde": lapl_gauss_not_dict_kde_pdf,
                 "lapl_gauss_not_dict_selected_densities" : selected_densities_lapl_gauss_not_dict,
                 "lapl_gauss_not_dict_mle_bic_model": lapl_gauss_not_dict_mle_bic_model,
                 "lapl_gauss_not_dict_mle_time": lapl_gauss_not_dict_time_mle,
                 "lapl_gauss_not_dict_mle_bic_time": lapl_gauss_not_dict_time_mle_bic,
                 "lapl_gauss_not_dict_kde_time": lapl_gauss_not_dict_time_kde,
                 "lapl_gauss_not_dict_kde_sj_time": lapl_gauss_not_dict_time_kde_sj,
                 "lapl_gauss_not_dict_ad_time": lapl_gauss_not_dict_time_ad,

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
    scales =[0.05, 0.1, 0.2, 0.5, 1]
    nodes_lapl = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for m in nodes_lapl:
        for scale in scales:
            densities.append(laplace(loc=m, scale=scale))
    #50 uniform densities 
    size_uniform = 10
    for i in range(size_uniform):
        densities.append(uniform_nonzero(i*1./size_uniform, 1./size_uniform))

    for N in [100, 500, 1000]:
        p = Pool(processes=9) 
        i=0
        #We send a batch of 20 tasks
        while i <=200:
            res = [p.apply_async(simu, args=(0, N)) for _ in range(20)]
            i += sum([r.get() for r in res])
        p.close()
        p.join()        