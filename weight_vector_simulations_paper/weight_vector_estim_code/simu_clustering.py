from datetime import datetime
import os
from multiprocessing import Pool
import pickle
from tools import l2_norm
from densitiesGenerator import  DensityGenerator

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
SELECT_THRESHOLD = 10e-5
MAX_COMPONENTS_MLE_BIC = 20
FOLDER = "dg_"+str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", ".") + "/"
os.makedirs(FOLDER)

def simu(N, K, dim):

    return 1

if __name__ == "__main__":
    print FOLDER
    for N in [100, 500, 1000]:
        for K in [2, 5, 10]:
            for dim in [2, 5, 10]:
                p = Pool(processes=9) 
                i=0
                #We send a batch of 20 tasks
                while i <=200:
                    res = [p.apply_async(simu, args=(N, K, dim)) for _ in range(20)]
                    i += sum([r.get() for r in res])
                p.close()
                p.join()        