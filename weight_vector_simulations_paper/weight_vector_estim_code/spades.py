from sklearn.base import BaseEstimator
import numpy as np
from numba import jit
from cvxpy import *
from multiprocessing import Pool
from queue import Queue

class SpadesEstimator(BaseEstimator):
    """
    from SPADES AND MIXTURE MODELS, by Florentina Bunea, Alexandre B. Tsybakov, Marten H. Wegkamp and Adrian Barbu.
    Another implementation using CVXPY for the minimization part.
    We use numba jit, if not present, comment the decorators @jit and imports.
    """

    def __init__(self, densities):
        self.SPADES_TRESH_ZERO = 1e-10
        self.ALPHA_GBM = 1e-3
        self.CV_SPLIT = 5
        self.INT_LINSPACE_PTS = 10000
        self.densities = densities
        self.K = len(densities)
        self.dens_matrix_int = self.dens_matrix_integral_gen()


    @jit
    def dens_matrix_f_x_gen(self, x_matrix):
        '''
        array of pdf(x) for each densities
        '''
        res = np.zeros([x_matrix.shape[0], self.K])
        for i in range(self.K):
            res[:,i] = self.densities[i].pdf(x_matrix)
        return res
    
    @jit
    def dens_matrix_integral_gen(self):
        X_int = np.linspace(0, 1, self.INT_LINSPACE_PTS)
        res = self.dens_matrix_f_x_gen(X_int)
        return res
    
    @jit
    def spades_solver(self, w_vect, f_X):
        """
        We use CVXPY for this minimization problem
        """
        lambda_ = Variable(self.K)
        norm2_f_sq = 1./self.INT_LINSPACE_PTS*sum_entries((self.dens_matrix_int*lambda_)**2)
        objective = Minimize(-2./f_X.shape[0]*sum_entries(f_X*lambda_) +
                             norm2_f_sq +
                             2*norm1(w_vect*lambda_))
        prob = Problem(objective)
        prob.solve()
        res = np.array(lambda_.value)
        #We zero all components lesser than the treshold
        res[res < self.SPADES_TRESH_ZERO] = 0
        return res

    def gbm(self, f_x):
        w_vect = -np.ones(self.K)
        w_vect[0] = 100
        w_vect[-1] = 0
        q = Queue()
        q.put((w_vect[0],w_vect[-1]))
        while not q.empty():
            a,b = q.get()
            w = 1.0*(a+b)/2
            lambd = self.spades_solver(w, f_x)
            k = len(lambd[lambd>0])-1
            if w_vect[k]==-1:
                w_vect[k]=w
            lambd = self.spades_solver(a, f_x)
            k_temp = len(lambd[lambd>0])-1
            if np.abs(k_temp-k) > 1 and np.abs(a-w) > self.ALPHA_GBM:
                q.put((a,w))
            lambd = self.spades_solver(b, f_x)
            k_temp = len(lambd[lambd>0])-1
            if np.abs(k_temp-k) > 1 and np.abs(b-w) > self.ALPHA_GBM:
                q.put((w,b))
        return w_vect
    
    def d_j_complement(self, X, j):
        mask = np.ones(X.shape, dtype=bool)
        mask[X.shape[0]/self.CV_SPLIT*j:X.shape[0]/self.CV_SPLIT*(j+1)] = False
        return X[mask]
    
    def gamma(self, X, lambd):
        return -2./X.shape[0]*(self.dens_matrix_f_x_gen(X).dot(lambd)).sum() + 1./self.INT_LINSPACE_PTS*(self.dens_matrix_int.dot(lambd)**2).sum()

    def w_select_cv(self, X):
        self.N = X.shape[0]
        #We split the dataset into self.CV_SPLIT parts D_j
        part_X = []
        for i in range(self.CV_SPLIT):
            part_X.append(X[self.N/self.CV_SPLIT*i : self.N/self.CV_SPLIT*(i+1)])
        #we create the complement for each j in range(self.CV_SPLIT), D_j := D\Dj
        part_X_comp = []
        for j in range(self.CV_SPLIT):
            part_X_comp.append(self.d_j_complement(X, j))
        L = np.zeros([self.K,self.CV_SPLIT])
        for k in range(self.K):
            for j in range(self.CV_SPLIT):
                F_X_j = self.dens_matrix_f_x_gen(part_X_comp[j])
                #Check if we have to extract k+1 or k
                w_k_j = self.gbm(F_X_j)[k]
                lambd = self.spades_solver(w_k_j, F_X_j)
                L[k,j] = self.gamma(part_X[j], lambd)
        self.L_final = 1./self.CV_SPLIT*L.sum(axis=1)
        self.k_final = np.argmin(np.array([self.L_final[i] +0.5*(i+1)*np.log(self.N)/self.N for i in range(self.K)]))
    
    def bbm(self,k):
        #TO CHECK
        w_vect = -np.ones(self.K)
        w_vect[0] = 100
        w_vect[-1] = 0
        q = Queue()
        q.put((w_vect[0],w_vect[-1]))
        while not q.empty():
            a,b = q.get()
            w = 1.0*(a+b)/2
            lambd = self.spades_solver(w, f_x)
            k = len(lambd[lambd>0])-1
            if w_vect[k]==-1:
                w_vect[k]=w
            lambd = self.spades_solver(a, f_x)
            k_temp = len(lambd[lambd>0])-1
            if np.abs(k_temp-k) > 1 and np.abs(a-w) > self.ALPHA_GBM:
                q.put((a,w))
            lambd = self.spades_solver(b, f_x)
            k_temp = len(lambd[lambd>0])-1
            if np.abs(k_temp-k) > 1 and np.abs(b-w) > self.ALPHA_GBM:
                q.put((w,b))
        return w_vect


    def fit(self, X):
        self.N = X.shape[0]
        print "Finding w"
        self.w = self.w_select_cv(X)
        self.lambd = self.spades_solver(self.w, self.dens_matrix_f_x_gen(X))



