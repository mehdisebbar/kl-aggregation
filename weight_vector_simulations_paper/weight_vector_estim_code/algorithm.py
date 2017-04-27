from sklearn.base import BaseEstimator
import numpy as np
from cvxpy import *


class WeightEstimator(BaseEstimator):

    def __init__(self, densities_dict, select_threshold=10e-10):
        self.densities_dict  = densities_dict
        self.K = len(self.densities_dict)
        self.select_threshold = select_threshold

    def fit(self, X):
        F = np.array([self.densities_dict[i].pdf(X) for i in range(self.K)]).T
        self.pi = Variable(self.K)
        constraints = [sum_entries(self.pi) == 1, self.pi >= 0]
        objective = Minimize(-sum_entries(log(F * self.pi)))
        prob = Problem(objective, constraints)
        #We try different solvers
        prob.solve()


    def select_densities(self):
        # return index and proba of selected densities according to select_threshold
        res = zip([i for i in range(self.K) if self.pi.value[i]>self.select_threshold],\
               (self.pi.value[self.pi.value>self.select_threshold]).tolist()[0])
        res.sort(key=lambda x: x[0])
        return res






