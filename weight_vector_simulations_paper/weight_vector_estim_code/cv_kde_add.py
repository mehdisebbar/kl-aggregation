"""
This code is an extension to add the KDE CV method, 
it is based on the python notebook without clean implementation
"""

from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import numpy as np
import uuid
from scipy.stats import multivariate_normal
from pythonABC.hselect import hsj
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from time import time



def cv_kde(X, n_pdf):
    """
    kde with CV bandwidth selection
    """
    a = time()
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.01, 1.0, 100)},
                        cv=20, n_jobs=8) # 20-fold cross-validation
    grid.fit(X[:, None])
    kde = grid.best_estimator_
    x_grid = np.linspace(0, 1, n_pdf)
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    b=time()
    return pdf, b-a

def data_extract_from_files(onlyfiles):
    for f in onlyfiles:
        yield pickle.load(open(folder+f)), f

def generate_data_tree(onlyfiles):
    data_tree = {}
    for k in [0, 10,50,100]:
        data_tree[k]={}
        for n in [100 , 500, 1000]:
            data_tree[k][n]=[]

    for f in onlyfiles:
        split = f.split('K')[1].split("N")
        for K in [0, 10,50,100]:
            if split[0]==str(K):
                K_ = K
        if split[1].startswith("500"):
            N_ = 500
        else :
            if pickle.load(open(folder+f))["rect_data"].shape[0] == 100:
                N_ = 100
            else:
                N_ = 1000
        data_tree[K_][N_].append(f)
    return data_tree

def construct_pdf_gmm(model, n_pdf):
    """
    Construct the pdf of the estimated Gaussian mixture
    """
    components = []
    for m, var in zip(model.means_, model.covariances_):
        components.append(multivariate_normal(m, var)) 
    return np.apply_along_axis(lambda x: model.weights_.dot(np.array([d.pdf(x) for d in components])), 0, np.linspace(0,1,n_pdf))   


def get_non_zero_weights(v):
    return np.array(zip(*[(idx, val) for (idx, val) in  enumerate(v) if val !=0]))

def results_extract(r):
    #return dict
    #
    densities = r["densities"]
    res = {}
    for type_dens in ["lapl_gauss", "uniform", "rect", "lapl_gauss_not_dict", "gauss"]:
        f_star = r[type_dens+"_f_star"]
        n_pdf = f_star.shape[0]
        x = np.linspace(0,1,n_pdf)
        X_ = r[type_dens+"_data"]
        #extract weight_estim
        res_weights_estim = r[type_dens+"_weight_vector_estim_lambda"]
        selected_densities_estim, weights_estim = get_non_zero_weights(res_weights_estim)
        f_weight_estim = np.apply_along_axis(lambda x: weights_estim.dot(np.array([densities[i].pdf(x) for i in selected_densities_estim.astype(int)])), 0, np.linspace(0,1,n_pdf))
        #removing non positive values:
        f_weight_estim[f_weight_estim <= 0] == 1e-20
        #extract adapative_dantzig
        lambda_adapative_dantzig = r[type_dens+"_adapative_dantzig"]
        f_adapative_dantzig = np.apply_along_axis(lambda x: lambda_adapative_dantzig.dot(np.array([d.pdf(x) for d in densities])), 0, np.linspace(0,1,n_pdf))        
        #removing non positive values:
        f_adapative_dantzig[f_adapative_dantzig <= 0] == 1e-20
        #Extract KDE
        f_kde= r[type_dens+"_kde"]
        #Extract KDE-hsj
        f_kde_hsj= r[type_dens+"_kde_hsj"]        
        #Perform kernel density estim CV
        f_kde_cv, kde_cv_time = cv_kde(X_, n_pdf)
        #Compute loss
        res[type_dens+"_L2_norm_weight_estim"] = 1./n_pdf*np.linalg.norm(f_star-f_weight_estim,axis=0)**2
        res[type_dens+"_KL_div_weight_estim"] = entropy(f_star, f_weight_estim)
        res[type_dens+"_L2_norm_adapative_dantzig"] = 1./n_pdf*np.linalg.norm(f_star-f_adapative_dantzig,axis=0)**2
        res[type_dens+"_KL_div_adapative_dantzig"] = entropy(f_star, f_adapative_dantzig)
        res[type_dens+"_L2_norm_kde"] = 1./n_pdf*np.linalg.norm(f_star-f_kde,axis=0)**2
        res[type_dens+"_KL_div_kde"] = entropy(f_star, f_kde)
        res[type_dens+"_L2_norm_kde_hsj"] = 1./n_pdf*np.linalg.norm(f_star-f_kde_hsj,axis=0)**2
        res[type_dens+"_KL_div_kde_hsj"] = entropy(f_star, f_kde_hsj)
        res[type_dens+"_L2_norm_kde_cv"] = 1./n_pdf*np.linalg.norm(f_star-f_kde_cv,axis=0)**2
        res[type_dens+"_KL_div_kde_cv"] = entropy(f_star, f_kde_cv)
        if type_dens == "cvx":
            cvx_lambda_fstar = cvx_lambda_f_star(len(densities), r["selected_densities"])
            res[type_dens+"_L2_lambda_adapative_dantzig"] = np.linalg.norm(cvx_lambda_fstar- r["cvx_adapative_dantzig"])
            res[type_dens+"_L2_lambda_weight_estim"] = np.linalg.norm(cvx_lambda_fstar- extract_lambda(r["cvx_weight_vector_estim_lambda"],len(densities)))
        #extract MLE GMM
        f_em_bic = construct_pdf_gmm(r[type_dens+"_mle_bic_model"], n_pdf=n_pdf)
        res[type_dens+"_KL_div_em_bic"] = entropy(f_star, f_em_bic)
        res[type_dens+"_L2_norm_em_bic"] = 1./n_pdf*np.linalg.norm(f_star-f_em_bic,axis=0)**2
        #temps
        res[type_dens+"_mle_time"] = r[type_dens+"_mle_time"]
        res[type_dens+"_kde_time"] = r[type_dens+"_kde_time"]
        res[type_dens+"_kde_sj_time"] = r[type_dens+"_kde_sj_time"]
        res[type_dens+"_kde_cv_time"] = kde_cv_time
        res[type_dens+"_ad_time"] = r[type_dens+"_ad_time"]
        res[type_dens+"_em_bic_time"] = r[type_dens+"_mle_bic_time"]
    return res

#cvx case:
def dataframe_gen(temp_res, type_f_star):
    data = {type_f_star+"_KL_div_weight_estim":[],
            type_f_star+"_KL_div_adapative_dantzig":[],
            type_f_star+"_KL_div_kde":[],
            type_f_star+"_KL_div_kde_hsj":[],
            type_f_star+"_KL_div_kde_cv":[],
            type_f_star+"_KL_div_em_bic":[],
            type_f_star+"_L2_norm_weight_estim":[],
            type_f_star+"_L2_norm_adapative_dantzig":[],
            type_f_star+"_L2_norm_kde":[],
            type_f_star+"_L2_norm_kde_hsj":[],
            type_f_star+"_L2_norm_kde_cv":[],
            type_f_star+"_L2_norm_em_bic":[],
            type_f_star+"_mle_time": [],
            type_f_star+"_kde_time": [],
            type_f_star+"_kde_sj_time": [],
            type_f_star+"_kde_cv_time": [],
            type_f_star+"_ad_time": [],
            type_f_star+"_em_bic_time": []
           }
    if type_f_star == "cvx":
        data["cvx_L2_lambda_weight_estim"] = [temp_res["cvx_L2_lambda_weight_estim"]]
        data["cvx_L2_lambda_adapative_dantzig"] = [temp_res["cvx_L2_lambda_adapative_dantzig"]]

    data[type_f_star+"_KL_div_weight_estim"].append(temp_res[type_f_star+"_KL_div_weight_estim"])
    data[type_f_star+"_KL_div_adapative_dantzig"].append(temp_res[type_f_star+"_KL_div_adapative_dantzig"])
    data[type_f_star+"_KL_div_kde"].append(temp_res[type_f_star+"_KL_div_kde"])
    data[type_f_star+"_KL_div_kde_hsj"].append(temp_res[type_f_star+"_KL_div_kde_hsj"])
    data[type_f_star+"_KL_div_kde_cv"].append(temp_res[type_f_star+"_KL_div_kde_cv"])
    data[type_f_star+"_KL_div_em_bic"].append(temp_res[type_f_star+"_KL_div_em_bic"])
    data[type_f_star+"_L2_norm_weight_estim"].append(temp_res[type_f_star+"_L2_norm_weight_estim"])
    data[type_f_star+"_L2_norm_adapative_dantzig"].append(temp_res[type_f_star+"_L2_norm_adapative_dantzig"])
    data[type_f_star+"_L2_norm_kde"].append(temp_res[type_f_star+"_L2_norm_kde"])
    data[type_f_star+"_L2_norm_kde_hsj"].append(temp_res[type_f_star+"_L2_norm_kde_hsj"])
    data[type_f_star+"_L2_norm_kde_cv"].append(temp_res[type_f_star+"_L2_norm_kde_cv"])
    data[type_f_star+"_L2_norm_em_bic"].append(temp_res[type_f_star+"_L2_norm_em_bic"])
    data[type_f_star+"_mle_time"].append(temp_res[type_f_star+"_mle_time"])
    data[type_f_star+"_kde_time"].append(temp_res[type_f_star+"_kde_time"])
    data[type_f_star+"_kde_sj_time"].append(temp_res[type_f_star+"_kde_sj_time"])
    data[type_f_star+"_kde_cv_time"].append(temp_res[type_f_star+"_kde_cv_time"])
    data[type_f_star+"_ad_time"].append(temp_res[type_f_star+"_ad_time"])
    data[type_f_star+"_em_bic_time"].append(temp_res[type_f_star+"_em_bic_time"])
    return pd.DataFrame(data)

def generate_result_dataframe(data_tree):
    df_results = None
    i = 0
    for K in [0]:
        for N in [100, 500, 1000]:
            object_list = data_extract_from_files(data_tree[K][N])
            for r, f in object_list:
                res_extract = results_extract(r)
                res = None
                for type_f_star in ["lapl_gauss", "uniform", "rect", "lapl_gauss_not_dict", "gauss"]:
                    df_extract_temp = dataframe_gen(res_extract, type_f_star)
                    if type(res) == pd.core.frame.DataFrame:
                        res = res.join(df_extract_temp)
                    else:
                        res = df_extract_temp
                res = res.join(pd.DataFrame({"K":[K],"N":[N], "file":f}))
                if type(df_results) == pd.core.frame.DataFrame:
                        df_results = df_results.append(res)
                else:
                        df_results = res
                i+=1
                print "file: ", i
    return df_results

def retrieve_type(row):
    type_row=row["Type"]
    type_dens = "None"
    method = "None"
    if "KL" in type_row:
        metric = "KL"
    else:
        metric = "L2"
    for t in ["uniform", "rect", "gauss", "lapl_gauss", "lapl_gauss_not_dict"]:
        if t in type_row:
            type_dens = t
    if ("weight_estim" in type_row) or ("_mle_" in type_row):
            method = "MLE"
    if ("adapative_dantzig" in type_row) or ("_ad_" in type_row) :
            method = "A.D"
    if "kde" in type_row:
            if "sj" in type_row:
                method = "KDE SJ"
            elif "cv" in  type_row:
                method = "KDE CV"
            else:
                method = "KDE"
    if "bic" in type_row:
            method = "EM-BIC"
    return pd.Series({"metric":metric, "type_dens":type_dens, "method": method})
    
def main():
    "bizarre ce unamed: 0 qui apparait"
    data_tree = generate_data_tree(onlyfiles)
    df_results = generate_result_dataframe(data_tree)    
    df_results.to_csv("./full_results_N_100_500_1000_"+extension_name+"_"+simu_folder+".csv")
    df_without_times = df_results[[c for c in df_results.columns if "time" not in c]]
    df3 = pd.melt(df_without_times.drop(["K","file"], axis=1), id_vars=["N"], var_name="Type", value_name="Loss")
    df5 = pd.concat([df3,df3.apply(retrieve_type, axis=1)], axis=1)
    df5.to_csv("./cleaned_results_N_100_500_1000_"+extension_name+"_"+simu_folder+".csv")

if __name__ == '__main__':
    simu_folder = "2017-06-12_18.37.51"
    folder = "/Users/mehdi/Downloads/"+simu_folder+"/"
    onlyfiles = [f for f in listdir(folder) if (isfile(join(folder, f)) and f.startswith("res_K"))]
    extension_name = "GLU"
    main()