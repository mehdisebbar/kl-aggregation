from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator

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
    
    def __init__(self, max_pca_comp = 2, kmeans_k = 5, subspace_cluster_dim = 2):
        self.max_pca_comp = max_pca_comp
        self.kmeans_k = kmeans_k
        self.subspace_cluster_dim = subspace_cluster_dim
        self.sc = StandardScaler()
        self.kmeans = KMeans(self.kmeans_k)

    def select_eigvals(self, eigval):
        #select singular values according to 
        #"The Optimal Hard Threshold for Singular Values is 4/sqrt(3)",  Matan Gavish and David L. Donoho
        #returns 2 components and their original indexes.
        res = []
        idx_selected = []
        sorted_eigval = sorted(eigval)[::-1]
        for idx, val in enumerate(sorted_eigval):
            if val > 2.858*np.median(eigval):
                res.append(val)
                idx_selected.append(idx)
        #We make sure to return at least 2 components
        if len(res) == 0:
            res = sorted_eigval[:2]
        if len(res) < 2:
            res=res+sorted_eigval[idx_selected[-1]+1:2]
        return res, [np.argwhere(eigval==e)[0][0] for e in res]
    
    def extract_principal_components(self, X):
        #we rescale the data before PCA
        X_pca = self.pca.fit_transform(self.sc.fit_transform(X))
        return X_pca
    
    def get_clusters(self, X2, X):
        densities = []
        self.kmeans.fit(X2)
        labels_ = self.kmeans.labels_
        means, covars = self.build_means_covars_from_labels(X, labels_)
        for j in range(self.kmeans_k):
            densities.append(self.build_normal_distributions(means[j], covars[j]))
        return densities
    
    def build_means_covars_from_labels(self, X, y):
        means = np.zeros([self.kmeans_k, self.p])
        covars = np.zeros([self.kmeans_k, self.p, self.p])
        for i in range(self.kmeans_k):
            means[i] = X[y==i].mean(axis = 0)
            covars[i] = np.cov((X[y==i]).T)
        return means, covars
    
    def build_normal_distributions(self, mean, covar):
        return multivariate_normal(mean, covar)
    
    def fit(self, X):
        self.densities = []
        self.p = X.shape[1]
        eigval, eigvect =  np.linalg.eig(1./X.shape[0]*X.T.dot(X))
        self.pca = PCA(self.p)
        X_pca = self.extract_principal_components(X)
        X2 = X[:, self.select_eigvals(eigval)[1]]
        for components_couple in combinations(range(X2.shape[1]), self.subspace_cluster_dim):
            self.densities.append(self.get_clusters(X[:, components_couple], X))
    
    def fit_transform(self, X):
        self.fit(X)
        return [item for sublist in self.densities for item in sublist]