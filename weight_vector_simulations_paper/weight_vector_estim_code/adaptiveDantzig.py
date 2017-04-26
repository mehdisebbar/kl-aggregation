from sklearn.base import BaseEstimator
import numpy as np
from numba import jit
from cvxpy import *
from scipy.integrate import quad
from time import time

class AdaptiveDantzigEstimator(BaseEstimator):

    def __init__(self, densities):
        self.densities = densities
        self.K = len(densities)
        self.dens_inf_norm = np.array([d.pdf(0) for d in self.densities])
        self.gram_matrix = self.gram_matrix_gen()
    
    def dens_matrix_f_x_gen(self, x_matrix):
        '''
        array of pdf(x) for each densities
        '''
        res = np.zeros([self.N, self.K])
        for i in range(self.K):
            res[:,i] = self.densities[i].pdf(x_matrix)
        return res
    
    def beta_hat_gen(self, x_matrix):
        '''
        \hat\Beta_m generation
        '''
        dens_matrix_f_x = self.dens_matrix_f_x_gen(x_matrix)
        return 1./self.N*dens_matrix_f_x.sum(axis=0)
    
    @jit
    def sigma_m_sq_gen(self, m):
        r = 0
        for i in range(self.N):
            for j in range(i):
                r+= (self.dens_matrix_f_x[:,m][i]-self.dens_matrix_f_x[:,m][j])**2
        return 1./(self.N*(self.N-1))*r
    
    def sigma_sq_gen(self):
        return np.array([self.sigma_m_sq_gen(m) for m in range(self.K)])
    
    @jit
    def sigma_m_tilde_gen(self, m, sigma_sq_m):
        a = 2*self.dens_inf_norm[m]*np.sqrt(2./self.N*sigma_sq_m*1.01*np.log(self.K))
        b = 8./(3*self.N)*self.dens_inf_norm[m]**2*1.01*np.log(self.K)
        return sigma_sq_m + a + b
    
    def sigma_tilde_gen(self):
        sigma_sq = self.sigma_sq_gen()
        return np.array([self.sigma_m_tilde_gen(m, sigma_sq[m]) for m in range(self.K)])

    @jit
    def eta_gamma_m(self, m):
        sigma = self.sigma_tilde_gen()
        a = np.sqrt(1./self.N*2.*sigma[m]*1.01*np.log(self.K))
        b = 1./(3*self.N)*2*self.dens_inf_norm[m]*1.01*np.log(self.K)
        return a+b
    
    def eta_gamma_gen(self):
        return np.array([self.eta_gamma_m(m) for m in range(self.K)])

    #@jit()
    def couple_dens(self, x,i,j):
        return self.densities[i].pdf(x)*self.densities[j].pdf(x)
    
    #@jit
    def gram_matrix_gen(self):
        G = np.zeros([self.K, self.K])
        for i in range(self.K):
            for j in range(self.K):
                G[i][j] = quad(self.couple_dens, 0, 1, args=(i, j))[0]
        return G
    
    def dantzig_estim_cmp(self, x_matrix):
        beta_hat = self.beta_hat_gen(x_matrix)
        eta_gamma = self.eta_gamma_gen()
        lambda_ = Variable(self.K)
        constraints = [abs((self.gram_matrix*lambda_)[m]-beta_hat[m]) <= eta_gamma[m] for m in range(self.K)]
        prob = Problem(Minimize(norm1(lambda_)), constraints)
        prob.solve()
        return np.array(lambda_.value).flatten()

    def fit(self, x_matrix):
        self.N = x_matrix.shape[0]
        self.dens_matrix_f_x = self.dens_matrix_f_x_gen(x_matrix)
        return self.dantzig_estim_cmp(x_matrix)

