import numpy as np
from scipy.stats import uniform

class DensityGenerator(object):
    """
    Density generator Class
    """
    def __init__(self, n_pdf = 10000, xmin=0, xmax=1):
        self.n_pdf = n_pdf
        self.xmin = xmin
        self.xmax = xmax

    def generate_uniform(self, n_points):
        X_ = uniform.rvs(size=n_points)
        return X_, np.apply_along_axis(uniform.pdf, 0, np.linspace(self.xmin, self.xmax, self.n_pdf))
    
    def generate_rect(self, n_points, dist_rect):
        """
        Generate a sample according to a mix of uniform segments
        Example of dist_rect:
        dist_rect = {
            (0,1./5) : 10./7,
            (1./5,2./5) : 5./7,
            (2./5,3./5) : 10./7,
            (3./5,4./5) : 0,
            (4./5,1) : 10./7
        }
        """
        X_ = np.linspace(self.xmin, self.xmax, n_points)
        proba = np.array([self.prob_estim(x, dist_rect) for x in X_])
        proba = proba / proba.sum()
        return np.random.choice(X_, size=n_points, p=proba), np.array(
            [self.prob_estim(x, dist_rect) for x in np.linspace(self.xmin, self.xmax, self.n_pdf)])
    
    def gaussian(self, n_points, densities=None, selected_densities=None, s=5, cvx_rand=False, weights=None):
        if densities == None:
            raise ValueError("Densities were not given")
        # If selected_densities is None, we select randomly s elements in densities, 
        # then we have two possibilities depending on cvx_rand:
        # false: equal weight for each densities, w=1/s
        # true: random weights
        if selected_densities == None:
            selected_densities = np.random.choice(len(densities), s, replace=False)
        s = len(selected_densities)
        if cvx_rand:
            # We generate the weights
            weights = np.random.randint(n_points * 100, size=(1, s))[0]
            weights = 1. * weights / weights.sum()
            sample_repartition_among_clusters = np.random.multinomial(n_points, weights, size=1)[0]
            return self.generate_points(n_points, s, densities, selected_densities, sample_repartition_among_clusters, weights), weights
        else:
            if weights==None:
                a = round(n_points/ s) * np.ones(s)
                # We adjust the size to the last element
                a[-1] = a[-1] - (a.sum() - n_points)
                sample_repartition_among_clusters = a.astype(int)
                weights = a/a.sum()
            else:
                weights = np.array(weights)
                sample_repartition_among_clusters = np.random.multinomial(n_points, weights, size=1)[0]
            return self.generate_points(n_points, s, densities, selected_densities, sample_repartition_among_clusters, weights)
    
    def generate_points(self, N, s, densities, selected_densities, sample_repartition_among_clusters, weights):
        X = np.array([])
        t = 0
        # We generate the sample according to the selected densities and the weights
        for i in range(s):
            X = np.hstack((X, densities[selected_densities[i]].rvs(sample_repartition_among_clusters[i])))
            np.random.shuffle(X)
        return (X, 
        np.apply_along_axis(lambda x: weights.dot(np.array([densities[i].pdf(x) for i in selected_densities])), 0, np.linspace(self.xmin, self.xmax, self.n_pdf)), 
        weights, 
        selected_densities)
    
    def prob_estim(self, x, dist_rect):
        for intval in sorted(dist_rect.keys()):
            if x <= intval[1]:
                return dist_rect[intval]
        return 0
    
            