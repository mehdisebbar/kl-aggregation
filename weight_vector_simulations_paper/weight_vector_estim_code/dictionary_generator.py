from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from tools import mle_bic
from copy import deepcopy
from tools import goodness_fit_densities

class DictionaryGenerator(BaseEstimator):
    """
    Generate a dictionary of gaussian densities by performing a PCA of 
    max_pca_comp components. On each subspace of dimension subspace_cluster_dim 
    spanned by each subspace_cluster_dim - combinations of principal components
    we perform a Kmeans clustering and recover the covariances and means in the 
    original spaces associated to these clusters. Finaly, We construct the 
    dictionary of normals with these means and covariances.
    TODO: a goodness-of-fit test to select the clusters that belong to the 
    same distribution.

    max_pca_comp = Number of principal components selected for PCA
    kmeans_k = Number of clusters on each subspaces
    subspace_cluster_dim = dimension of subspace spanned by principal comps.
    """
    
    def __init__(self, max_pca_comp = 2, kmeans_k = 10, subspace_cluster_dim = 2, verbose=False, pc_select = False, gof = False):
        self.pc_select = pc_select
        self.max_pca_comp = max_pca_comp
        self.kmeans_k = kmeans_k
        self.subspace_cluster_dim = subspace_cluster_dim
        self.sc = StandardScaler()
        self.kmeans = KMeans(self.kmeans_k)
        self.verbose = verbose
        self.gof = gof
    
    @staticmethod
    def tau(beta):
        #Approximation of singular value threshold from Gavish & Donoho (2014)
        return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43

    def select_eigvals(self, eigval):
        #select singular values according to 
        #"The Optimal Hard Threshold for Singular Values is 4/sqrt(3)",  Matan Gavish and David L. Donoho (2014)
        #returns at least 2 components and their original indexes.
        res = []
        idx_selected = []
        sorted_eigval = sorted(eigval)[::-1]
        for idx, val in enumerate(sorted_eigval):
            if val > self.tau(self.beta)*np.median(eigval): #(eq4) from Gavish.Donoho 
                res.append(val)
                idx_selected.append(idx)
        #We make sure to return at least 2 components
        if len(res) == 0:
            res = sorted_eigval[:2]
        if len(res) < 2:
            res=res+sorted_eigval[idx_selected[-1]+1:2]
        return res, [np.argwhere(eigval==e)[0][0] for e in res]
    
    def get_clusters(self, X2, X, method="em-bic"):
        """
        Get clusters in the subspace. Default method is EM-bic with kmeans_k 
        maximum clusters, otherwise KMeans with kmeans_k clusters.
        X2: projected data into the subspace
        X: original data, used to recover full variances and means.
        """
        if method == "em-bic":
            _, model = mle_bic(X2, self.kmeans_k)
            labels_ = model.predict(X2)
            self.k_ = model.get_params()['n_components']
            if self.verbose:
                print "selected clusters with em-bic: ", self.k_
        else:
            self.kmeans.fit(X2)
            labels_ = self.kmeans.labels_
            self.k_ = self.kmeans_k
        densities = []
        means, covars = self.build_means_covars_from_labels(X, labels_)
        for j in range(self.k_):
            densities.append(self.build_normal_distributions(means[j], covars[j]))
        return densities
    
    def build_means_covars_from_labels(self, X, y):
        means = np.zeros([self.k_, self.p])
        covars = np.zeros([self.k_, self.p, self.p])
        for i in range(self.k_):
            means[i] = X[y==i].mean(axis = 0)
            covars[i] = np.cov((X[y==i]).T)
        return means, covars
    
    def build_normal_distributions(self, mean, covar):
        return multivariate_normal(mean, covar, allow_singular=True)
    
    def fit(self, X):
        self.densities = []
        self.p = X.shape[1]
        #beta for Gavish and Donoho 2014
        self.beta = X.shape[1]/X.shape[0]
        X_std = self.sc.fit_transform(X)
        if self.pc_select:
            _, s, _ = np.linalg.svd(X_std)
            selected_eigvals = self.select_eigvals(s)[1]
        else:
            selected_eigvals = range(self.p)
        if self.verbose:
            print "Selected eigenvalues: ", selected_eigvals
        self.pca = PCA(len(selected_eigvals))
        X_pca = self.pca.fit_transform(X_std)
        for components_couple in combinations(range(X_pca.shape[1]), self.subspace_cluster_dim):
            self.densities.append(self.get_clusters(X[:, components_couple], X))
    
    def fit_transform(self, X):
        self.fit(X)
        self.densities_flatten = [item for sublist in self.densities for item in sublist]
        return self.densities_flatten

    def simplify_gof(self):
        """
        Perform the goodness-of-fit test from tools, and return a smaller densities list.
        """
        if hasattr(self, 'densities'):
            self.densities_flatten = [item for sublist in self.densities for item in sublist]
            densities_gof_simplified = deepcopy(self.densities_flatten)
            for i in range(len(densities_gof_simplified)):
                for j in range(i, len(densities_gof_simplified)):
                    if densities_gof_simplified[j]!= None and densities_gof_simplified[i]!= None:
                        #we call goodness_fit_densities() here, returns a  list of unique densities according
                        #to the test
                        if len(goodness_fit_densities([densities_gof_simplified[i], densities_gof_simplified[j]])) == 1 and i!=j:
                            densities_gof_simplified[j]=None
            return [d for d in densities_gof_simplified if d!= None]
        else:
            print "Execute fit_transform before."
            return None
    