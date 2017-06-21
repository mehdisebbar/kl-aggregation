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
    
    def __init__(self, max_pca_comp = 2, kmeans_k = 2, subspace_cluster_dim = 2):
        self.max_pca_comp = max_pca_comp
        self.kmeans_k = kmeans_k
        self.subspace_cluster_dim = subspace_cluster_dim
        self.pca = PCA(self.max_pca_comp)
        self.sc = StandardScaler()
        self.kmeans = KMeans()
    
    def extract_principal_components(self, X):
        #we rescale the data before PCA
        X_pca = self.pca.fit_transform(self.sc.fit_transform(X))
        return X_pca
    
    def get_clusters(self, X_pca, X):
        densities = []
        self.kmeans.fit(X_pca)
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
        X_pca = self.extract_principal_components(X)
        for components_couple in combinations(range(X_pca.shape[1]), self.subspace_cluster_dim):
            self.densities.append(self.get_clusters(X[:, components_couple], X))
    
    def fit_transform(self, X):
        self.fit(X)
        return [item for sublist in self.densities for item in sublist]