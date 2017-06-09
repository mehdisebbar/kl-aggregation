from sklearn.base import BaseEstimator
import numpy as np
from cvxpy import *
from numba import jit
from tools import simplex_proj_numba

class WeightEstimator_cvxpy(BaseEstimator):

    def __init__(self, densities_dict, select_threshold=1e-5):
        self.densities_dict  = densities_dict
        self.K = len(self.densities_dict)
        self.select_threshold = select_threshold

    def fit(self, X):
        F = np.array([self.densities_dict[i].pdf(X) for i in range(self.K)]).T
        self.pi = Variable(self.K-1)
        constraints = [sum_entries(self.pi) <= 1, self.pi >= 0]
        f = -sum_entries(log(F[:,:-1] * self.pi+F[:,-1]*(1-sum_entries(self.pi))))
        objective = Minimize(f)
        prob = Problem(objective, constraints)
        prob.solve()
        temp_res = [v.value for v in self.pi]
        self.pi_final = np.array(temp_res+[1-sum(temp_res)])
        self.prob = prob

    def select_densities(self, select_threshold=1e-5):
        # return index and proba of selected densities according to select_threshold
        res = zip([i for i in range(self.K) if self.pi_final[i]>self.select_threshold],\
               (self.pi_final[self.pi_final>self.select_threshold]).tolist())
        res.sort(key=lambda x: x[0])
        return res

class WeightEstimator(BaseEstimator):
    """
    Finds the best convex combination of densities in dict by minimizing the 
    log likelihood. Based on FISTA.abs
    """

    def __init__(self, densities_dict, select_threshold=1e-5):
        self.densities_dict  = densities_dict
        self.K = len(self.densities_dict)
        self.select_threshold = select_threshold
        self.h = 1e-3
        self.eps = 1e-5 #treshold on fista
    
    def fit(self, X):
        self.N = X.shape[0]
        F= np.zeros([self.N ,self.K])
        for i in range(self.K):
            F[:,i] = self.densities_dict[i].pdf(X)        
        self.pi_final = self.fista_code(F, X, self.eps)
        
    @jit()
    def fista_code(self, F, X, eps):
        pi_prev = np.ones(self.K)/self.K
        alpha_prev = np.ones(self.K)/self.K
        alpha_next = 10*alpha_prev
        t_prev = 1
        while np.linalg.norm(alpha_prev-alpha_next)> eps:
            alpha_prev = alpha_next
            alpha_next = simplex_proj_numba(pi_prev-self.h*self.gradient(F, pi_prev))
            t_next = (1. + np.sqrt(1+4*t_prev**2))/2
            pi_next = alpha_next + (t_prev - 1)/t_next * (alpha_next - alpha_prev)
            pi_prev = pi_next
            t_prev = t_next
        return alpha_next
    
    @jit()
    def gradient(self, F, pi):
        return -1./self.N*(F/((pi*F).sum(axis=1).reshape(-1,1))).sum(axis=0)


