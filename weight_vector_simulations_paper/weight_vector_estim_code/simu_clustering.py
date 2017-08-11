from datetime import datetime
import os
from multiprocessing import Pool
import pickle
from time import time
import uuid
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from tools import kl_norm, l2_norm, GaussianMixtureGen, mle_bic, KdeCV
from densitiesGenerator import  DensityGenerator
from algorithm import WeightEstimator
from dictionary_generator import DictionaryGenerator
from pypmc.density.mixture import create_gaussian_mixture
from numpy.random import seed

"""
We compare the KL-aggregation density estimator with the EM-BIC density estimator
on a problem of clustering/density estimation.
From a sample drawn from a p dimensional Gaussian mixture ok K components, we
generate a dictionary of densities via DensityGenerator class. It performs PCA and
runs kmeans on points projected on all subspaces spanned by 2 components of the PCA.
We runs after the KL-aggregator on those densities and EM+BIC on the original space.
Finally, KL and L2 norms are computed via Importance sampling. Computing times are 
recorded.
"""

N_PDF = 10000
KMEANS_K = 10
MAX_COMPONENTS_MLE_BIC = 10
SAMPLE_SIZE = 10000
MAX_EM_BIC_K = 20
N_JOBS = 8
HYPERCUBE_SIZE = 3
FOLDER = "dg_"+str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", ".") + "/"
os.makedirs(FOLDER)

#def goodness_fit_densities(densities_list, SAMPLE_SIZE=100):
#    new_dens = []
#    for d1, d2 in list(itertools.combinations(densities_list, 2)):
#        s = 0
#        #This might not be statistically correct, but helps to smooth the result, it 
#        for _ in range(10):
#            X1 = np.hstack([d1.rvs(SAMPLE_SIZE), np.ones(SAMPLE_SIZE).reshape(-1,1), np.arange(SAMPLE_SIZE).reshape(-1,1)])
#            X2 = np.hstack([d2.rvs(SAMPLE_SIZE), -np.ones(SAMPLE_SIZE).reshape(-1,1), np.arange(SAMPLE_SIZE, 2*SAMPLE_SIZE).reshape(-1,1)])
#            X = np.vstack([X1, X2])
#            np.random.shuffle(X)
#            indexes = X[:,-1]
#            X = X[:,:-1]
#            clf = LogisticRegression()
#            #clf = RandomForestClassifier(max_depth=5, n_estimators=20)
#            clf.fit(X[:,:-1], X[:,-1])
#            scores = clf.predict_proba(X[:,:-1])
#            s_plus = scores[indexes < SAMPLE_SIZE][:,0]
#            s_min = scores[indexes >= SAMPLE_SIZE][:,0]
#            a = ks_2samp(s_plus, s_min)
#            if a.pvalue > 0.05:
#                s+=1
#        if s > 0.5 and d1 not in new_dens:
#            new_dens.append(d1)
#        else:
#            if d1 not in new_dens:
#                new_dens.append(d1)
#            if d2 not in new_dens:
#                new_dens.append(d2)
#    return new_dens

class KLaggDensity(object):
    """
    Class of the mixture density generated by th KL-aggreg algorithm
    """
    def __init__(self, weights, density_dict):
        self.weights = weights
        self.density_dict = density_dict
    
    def pdf(self, x):
        return  self.weights.dot(np.array([d.pdf(x) for d in self.density_dict]))

class GaussMixtureDensity(object):
    """
    General class for a Gaussian mixture density given weights, centers and cov
    """
    def __init__(self, weights, centers, cov):
        self.weights = weights
        self.centers = centers
        self.cov = cov
    
    def pdf(self, x):
        return self.weights.dot(np.array([multivariate_normal(self.centers[i], self.cov[i]).pdf(x) for i in range(len(self.weights))]))


class IntegrandL2Density(object):
    """
    Compute the integrand (1-g/f)^2*f, f sampling distrib known, g estimator of density.
    """
    def __init__(self, f, g):
        """
        f and g are pdf functions
        """
        self.f = f
        self.g = g

    def pdf(self, x):
        return (1-self.g(x)/self.f(x))**2*self.f(x)

class IntegrandKLDensity(object):
    """
    Compute the integrand log(f/g), f sampling distrib known, g estimator of density.
    """
    def __init__(self, f, g):
        """
        f and g are pdf functions
        """
        self.f = f
        self.g = g

    def pdf(self, x):
        return np.log(self.f(x)/self.g(x))

class BasicGen(object):
    #dirty gross generator
    def __init__(self, dim=5):
        self.dim = dim
        self.params = [
            (np.array([[3, 0, 0, 0, 0],
                      [0, 0.1, 0, 0, 0],
                      [0, 0, 0.1, 0, 0],
                      [0, 0, 0, 0.1, 1],
                      [0, 0, 0, 1, 3]]), 
             np.array([0.1, 0.1, 0.1, 0.1, 0.1])), 
            (np.array([[0.1, 0, 0, 0, 0],
                            [0, 2, -0.2, 0, 0],
                            [0, -0.2, 0.1, 0, 0],
                            [0, 0, 0, 0.1, 0],
                            [0, 0, 0, 0, 1]]),
             np.array([0.8, 0.8, 0.8, 0.8, 0.8])), 
            (np.array([[1, 0.1, 1.9, 0, 0],
                       [0.1, 0.5, -1, 0, 0],
                       [1.9, -1, 3, 0, 0],
                       [0, 0, 0, 0.1, 0],
                       [0, 0, 0, 0, 0.1]]),
             np.array([0.1, 0.4, 0.6, 0.8, 0.1])),
            (np.array([[0.5, 1, -1, 0, 0],
                       [1, 2, 0, 0, 0],
                       [-1, 0, 0.1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0.1]]),
             np.array([0.5, 0.8, 0.4, 0.4, 0.4])),
            (np.array([[4, 0, 0, 0, 0],
                       [0, 0.5, 0, 0, 0],
                       [0, 0, 0.2, 0, 0],
                       [0, 0, 0, 0.2, 0],
                       [0, 0, 0, 0, 0.2]]),
             np.array([0.9, 0.2, 0.9, 0.9, 0.9])),
            (np.array([[0.1, 0, -1, 0, 0],
                       [0,  6, 0,   0, 0],
                       [-1, 0, 0.1, 0, 0],
                       [0, 0, 0, 0.1, 0],
                       [0, 0, 0, 0, 0.1]]),
             np.array([1, 0.8, 0.2, 0.4, 0.4]))
        ]
        self.change_dim(self.dim)
        self.means, self.variances = zip(*self.params)
    
    def change_dim(self, p):
        params = []
        for cov, m in self.params:
            C = cov[:p,:p]
            params.append((m[:p], 1e-3*C.T.dot(C)))
        self.params = params
    
    def get_params(self):
        return self.means, self.variances
    
    def sample(self, N, with_ids = False):
        #We generate a dataset with specific data
        K = len(self.params)
        X = multivariate_normal(self.params[0][0], self.params[0][1]).rvs(N/K)
        ids = np.ones(N/K)
        i = 2
        for m, cov in self.params[1:]:
            X = np.vstack([X, multivariate_normal(m, cov).rvs(N/K)])
            ids = np.hstack([ids, i*np.ones(N/K)])
            i+=1
        if with_ids:
            X = np.hstack([X, ids.reshape(-1,1)])
        np.random.shuffle(X)
        return X

def simu(N, K, dim, gof_test= True, pc_select = True, write_results = True, verbose = False):
    #reset the random generator, useful for multiprocess 
    seed()
    try:
        if verbose:
            print "init N=",N," dim=",dim
        #Some initialization
        #sc = StandardScaler()
        max_pca_comp = dim/2+1
        # We generate the Gaussian mixture
        #gg = GaussianMixtureGen(dim, weights)
        #centers_star, cov_star = gg.get_params()
        gg = BasicGen(dim)
        centers_star, cov_star = gg.get_params()
        #X_ = gg.sample(N)
        X_ids = gg.sample(N, with_ids=True)
        ids = X_ids[:,-1]
        X_ = X_ids[:,:-1]
        K = len(set(ids))
        weights = 1./K*np.ones(K)
        #We normalize the data for the PCA in the KL aggreg
        #X = sc.fit_transform(X_)
        X = X_
        #We generate the target density f_star from the components
        f_star = GaussMixtureDensity(weights, centers_star, cov_star)
        f_star_sampling = create_gaussian_mixture(centers_star, cov_star, weights)
        ######################
        # KL-AGGREG. ALGORITHM
        ######################
        if verbose:
            print "starting KL-aggreg"
        time_kl_aggreg_start = time()
        dg = DictionaryGenerator(kmeans_k=KMEANS_K, max_pca_comp=max_pca_comp, subspace_cluster_dim=2, pc_select=pc_select)
        X_train_dict_gen = X[:N/2]
        X_train_kl_aggreg = X[N/2:] 
        if gof_test:
            dg.fit(X)
            densities_dict = dg.simplify_gof()
        else:
            densities_dict = dg.fit_transform(X)
        cl = WeightEstimator(densities_dict=densities_dict)
        cl.fit(X)
        time_kl_aggreg_stop = time()
        kl_aggreg_weights = cl.pi_final
        kl_aggreg_density = KLaggDensity(kl_aggreg_weights, densities_dict)
        #Compute L2 loss
        kl_aggreg_integrand_L2_loss = IntegrandL2Density(f_star.pdf, kl_aggreg_density.pdf)
        kl_aggreg_l2 = l2_norm(kl_aggreg_integrand_L2_loss.pdf, f_star_sampling, sample_size=SAMPLE_SIZE,  hypercube_size=HYPERCUBE_SIZE)
        #Compute KL loss
        kl_aggreg_integrand_KL_loss = IntegrandKLDensity(f_star.pdf, kl_aggreg_density.pdf)
        kl_aggreg_kl = kl_norm(kl_aggreg_integrand_KL_loss.pdf, f_star_sampling, sample_size=SAMPLE_SIZE,  hypercube_size=HYPERCUBE_SIZE)
        if verbose:
            print "KL-aggreg done"
            print "KL-loss", kl_aggreg_kl
            print "L2-loss", kl_aggreg_l2
        #################
        #EM-BIC ALGORITHM
        #################
        if verbose:
            print "starting EM-BIC"
        time_em_start = time()
        _, em_model = mle_bic(X, MAX_EM_BIC_K)
        time_em_stop = time()
        em_density = GaussMixtureDensity(em_model.weights_, em_model.means_, em_model.covariances_)
        #Compute L2 loss
        em_integrand_L2_loss = IntegrandL2Density(f_star.pdf, em_density.pdf)
        em_l2 = l2_norm(em_integrand_L2_loss.pdf, f_star_sampling, sample_size=SAMPLE_SIZE,  hypercube_size=HYPERCUBE_SIZE)
        #Compute KL loss
        em_integrand_KL_loss = IntegrandKLDensity(f_star.pdf, em_density.pdf)
        em_kl = kl_norm(em_integrand_KL_loss.pdf, f_star_sampling, sample_size = SAMPLE_SIZE,  hypercube_size=HYPERCUBE_SIZE)
        if verbose:
            print "EM-BIC done"
            print "KL-loss", em_kl
            print "L2-loss", em_l2
        #################
        #KDE-CV ALGORITHM
        #################
        if verbose:
            print "starting KDE-CV"
        kde = KdeCV(n_jobs = 1, cv=10, bw = np.linspace(0.01, 1.0, 20))
        time_kde_start = time()
        kde.fit(X)
        time_kde_stop = time()
        kde_integrand_KL_loss = IntegrandKLDensity(f_star.pdf, kde.pdf)
        kde_kl = kl_norm(kde_integrand_KL_loss.pdf, f_star_sampling, sample_size = SAMPLE_SIZE,  hypercube_size=HYPERCUBE_SIZE)   
        kde_integrand_L2_loss = IntegrandL2Density(f_star.pdf, kde.pdf)
        kde_l2 = l2_norm(kde_integrand_L2_loss.pdf, f_star_sampling, sample_size=SAMPLE_SIZE,  hypercube_size=HYPERCUBE_SIZE)  
        #Compute times
        kl_aggreg_time = time_kl_aggreg_stop-time_kl_aggreg_start
        em_bic_time = time_em_stop-time_em_start
        kde_cv_time = time_kde_stop-time_kde_start
        if verbose:
            print "KDE-CV done"
            print "KL-loss", kde_kl
            print "L2-loss", kde_l2
        #Writing results
        if write_results:
            print "OK, writing results"
            pickle.dump({"K" : K,
                         "p" : dim,
                         "N" : N,
                         "MLE_l2" : kl_aggreg_l2,
                         "MLE_KL" : kl_aggreg_kl,
                         "MLE_time" : kl_aggreg_time,
                         "EM_l2" : em_l2,
                         "EM_KL" : em_kl,
                         "EM_time" : em_bic_time,
                         "KdeCV_l2" : kde_l2,
                         "KdeCV_KL" : kde_kl,
                         "KdeCV_time" : kde_cv_time
                     }, open(FOLDER +
                             "res_" + "K" + str(K) + "p" + str(dim) + "N" + str(N) +"_"+ type_simu_to_str(gof_test, pc_select)+"_"+str(uuid.uuid4()), "wb"))
        else:
            # we print the results, for testing.
            print {"K" : K,
                         "p" : dim,
                         "N" : N,
                         "MLE_l2" : kl_aggreg_l2,
                         "MLE_KL" : kl_aggreg_kl,
                         "MLE_time" : kl_aggreg_time,
                         "EM_l2" : em_l2,
                         "EM_KL" : em_kl,
                         "EM_time" : em_bic_time,
                         "KdeCV_l2" : kde_l2,
                         "KdeCV_KL" : kde_kl,
                         "KdeCV_time" : kde_cv_time
                     }
        return 1
    except Exception as e:
        print e
        return 0
def type_simu_to_str(gof,pc):
    if gof:
        s1 = "gof"
    else:
        s1 = ""
    if pc:
        s2 = "pc"
    else:
        s2 = ""
    return s1+"_"+s2


def multiprocess_code(N_list, dim_list):     
    for dim in dim_list:
        for N in N_list:
            ##########################
            #gof = false, pc = false
            p = Pool(processes=N_JOBS) 
            i=0
            #We send a batch of 20 tasks
            while i <=100:
                res = [p.apply_async(simu, args=(N, 4, dim, False, False, True, True)) for _ in range(20)]
                i += sum([r.get() for r in res])
            p.close()
            p.join()
            ##########################       
            #gof = true, pc = true
            p = Pool(processes=N_JOBS) 
            i=0
            #We send a batch of 20 tasks
            while i <=100:
                res = [p.apply_async(simu, args=(N, 4, dim, True, True, True, True)) for _ in range(20)]
                i += sum([r.get() for r in res])
            p.close()
            p.join()   



if __name__ == "__main__":
    print FOLDER
    import argparse
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('N', type=str,
                    help='N')
    # Optional positional argument
    parser.add_argument('dim', type=str,
                    help='dimension split with ,')
    parser.add_argument('mp', type=str,
                    help='Multiprocessing')
    args = parser.parse_args()
    N_list = [int(item) for item in args.N.split(',')]
    dim_list = [int(item) for item in args.dim.split(',')]

    if args.mp == "True":
        multiprocess_code(N_list, dim_list)
    else:
        for _ in range(200):
            for dim in dim_list:
                for N in N_list:        
                    simu(N, 4, dim, False, False, True, True)
            
#    simu_list = [
#        (2,2,100),
#        (4,2,100),
#        (4,2,500),
#        (4,5,100),
#        (4,5,500),
#        (4,5,1000)
#        #(10,5,100),
#        #(10,5,500),
#        #(10,5,1000),
#        #(10,20,500),
#        #(10,20,1000)
#    ]
#   for K, dim, N in simu_list:
#       if K > 2**dim:
#           pass
#       else:
#           p = Pool(processes=8) 
#           i=0
#           #We send a batch of 20 tasks
#           while i <=100:
#               res = [p.apply_async(simu, args=(N, K, dim)) for _ in range(20)]
#               i += sum([r.get() for r in res])
#           p.close()
#           p.join()   
# 
