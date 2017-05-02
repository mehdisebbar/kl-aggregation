from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import numpy as np
from pythonABC.hselect import hsj
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import matplotlib.pyplot as plt
import uuid


def get_files(folder):
    onlyfiles = [f for f in listdir(folder) if (isfile(join(folder, f)) and f.startswith("res_K"))]
    return onlyfiles

def data_extract_from_files(onlyfiles, folder):
    for f in onlyfiles:
        yield pickle.load(open(folder+f)), f

def generate_data_tree(onlyfiles, folder):
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
            try:
                if pickle.load(open(folder+f))["rect_data"].shape[0] == 100:
                    N_ = 100
                else:
                    N_ = 1000
            except:
                pass
        data_tree[K_][N_].append(f)
    return data_tree



def results_extract(r):
    #return dict
    #
    densities = r["densities"]
    res = {}
    for type_dens in ["lapl_gauss", "uniform", "rect", "lapl_gauss_not_dict", "gauss"]:
        f_star = r[type_dens+"_f_star"]
        n_pdf = f_star.shape[0]
        x = np.linspace(0,1,n_pdf)
        #extract weight_estim
        res_weights_estim = r[type_dens+"_weight_vector_estim_lambda"]
        selected_densities_estim, weights_estim = np.array(zip(*res_weights_estim))
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
        #Perform kernel density estim
        res[type_dens+"_L2_norm_weight_estim"] = 1./n_pdf*np.linalg.norm(f_star-f_weight_estim,axis=0)**2
        res[type_dens+"_KL_div_weight_estim"] = entropy(f_star, f_weight_estim)
        res[type_dens+"_L2_norm_adapative_dantzig"] = 1./n_pdf*np.linalg.norm(f_star-f_adapative_dantzig,axis=0)**2
        res[type_dens+"_KL_div_adapative_dantzig"] = entropy(f_star, f_adapative_dantzig)
        res[type_dens+"_L2_norm_kde"] = 1./n_pdf*np.linalg.norm(f_star-f_kde,axis=0)**2
        res[type_dens+"_KL_div_kde"] = entropy(f_star, f_kde)
        res[type_dens+"_L2_norm_kde_hsj"] = 1./n_pdf*np.linalg.norm(f_star-f_kde_hsj,axis=0)**2
        res[type_dens+"_KL_div_kde_hsj"] = entropy(f_star, f_kde_hsj)
        if type_dens == "cvx":
            cvx_lambda_fstar = cvx_lambda_f_star(len(densities), r["selected_densities"])
            res[type_dens+"_L2_lambda_adapative_dantzig"] = np.linalg.norm(cvx_lambda_fstar- r["cvx_adapative_dantzig"])
            res[type_dens+"_L2_lambda_weight_estim"] = np.linalg.norm(cvx_lambda_fstar- extract_lambda(r["cvx_weight_vector_estim_lambda"],len(densities)))

    return res

def dataframe_gen(temp_res, type_f_star):
    data = {type_f_star+"_KL_div_weight_estim":[],
            type_f_star+"_KL_div_adapative_dantzig":[],
            type_f_star+"_KL_div_kde":[],
            type_f_star+"_KL_div_kde_hsj":[],
            type_f_star+"_L2_norm_weight_estim":[],
            type_f_star+"_L2_norm_adapative_dantzig":[],
            type_f_star+"_L2_norm_kde":[],
            type_f_star+"_L2_norm_kde_hsj":[]
            }
    if type_f_star == "cvx":
        data["cvx_L2_lambda_weight_estim"] = [temp_res["cvx_L2_lambda_weight_estim"]]
        data["cvx_L2_lambda_adapative_dantzig"] = [temp_res["cvx_L2_lambda_adapative_dantzig"]]

    data[type_f_star+"_KL_div_weight_estim"].append(temp_res[type_f_star+"_KL_div_weight_estim"])
    data[type_f_star+"_KL_div_adapative_dantzig"].append(temp_res[type_f_star+"_KL_div_adapative_dantzig"])
    data[type_f_star+"_KL_div_kde"].append(temp_res[type_f_star+"_KL_div_kde"])
    data[type_f_star+"_KL_div_kde_hsj"].append(temp_res[type_f_star+"_KL_div_kde_hsj"])
    data[type_f_star+"_L2_norm_weight_estim"].append(temp_res[type_f_star+"_L2_norm_weight_estim"])
    data[type_f_star+"_L2_norm_adapative_dantzig"].append(temp_res[type_f_star+"_L2_norm_adapative_dantzig"])
    data[type_f_star+"_L2_norm_kde"].append(temp_res[type_f_star+"_L2_norm_kde"])
    data[type_f_star+"_L2_norm_kde_hsj"].append(temp_res[type_f_star+"_L2_norm_kde_hsj"])
    
    return pd.DataFrame(data)

def generate_dataframe(data_tree, folder):
    df_results = None
    for K in [0]:
        for N in [100, 500, 1000]:
            object_list = data_extract_from_files(data_tree[K][N], folder)
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
    return df_results

def generate_plot(df_results):
    plt.subplots(figsize=(15,20))
    colors = ['lightblue', 'lightgreen', 'tan', 'pink']
    def boxplot_params(boxplotElements):
        for element in boxplotElements['medians']:
            element.set_color('red')
            element.set_linewidth(1)
        for element in boxplotElements['boxes']:
            element.set_linewidth(1)
            element.set_linestyle('-')
        for element in boxplotElements['whiskers']:
            element.set_color('red')
            element.set_linewidth(1)
        for element in boxplotElements['caps']:
            element.set_color('blue')


    ##### uniform
    ###############
    #KL div
    plt.subplot(421)
    boxplotElements = plt.boxplot([df_results['uniform_KL_div_kde'],
                 df_results['uniform_KL_div_kde_hsj'],
                 df_results['uniform_KL_div_adapative_dantzig'],
                 df_results['uniform_KL_div_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("uniform dict KL div")
    #L2 norm
    plt.subplot(422)
    boxplotElements = plt.boxplot([df_results['uniform_L2_norm_kde'],
                 df_results['uniform_L2_norm_kde_hsj'],
                 df_results['uniform_L2_norm_adapative_dantzig'],
                 df_results['uniform_L2_norm_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("uniform dict L2 norm")

    #####Lapl Gauss
    ###############
    #KL div
    plt.subplot(423)
    boxplotElements = plt.boxplot([df_results['lapl_gauss_KL_div_kde'],
                 df_results['lapl_gauss_KL_div_kde_hsj'],
                 df_results['lapl_gauss_KL_div_adapative_dantzig'],
                 df_results['lapl_gauss_KL_div_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("Lapl-Gauss dict KL div")
    #L2 norm
    plt.subplot(424)
    boxplotElements = plt.boxplot([df_results['lapl_gauss_L2_norm_kde'],
                 df_results['lapl_gauss_L2_norm_kde_hsj'],
                 df_results['lapl_gauss_L2_norm_adapative_dantzig'],
                 df_results['lapl_gauss_L2_norm_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("Lapl-Gauss dict L2 norm")

    ##### Gauss
    ###############
    #KL div
    plt.subplot(425)
    boxplotElements = plt.boxplot([df_results['gauss_KL_div_kde'],
                 df_results['gauss_KL_div_kde_hsj'],
                 df_results['gauss_KL_div_adapative_dantzig'],
                 df_results['gauss_KL_div_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("Gauss dict KL div")
    #L2 norm
    plt.subplot(426)
    boxplotElements = plt.boxplot([df_results['gauss_L2_norm_kde'],
                 df_results['gauss_L2_norm_kde_hsj'],
                 df_results['gauss_L2_norm_adapative_dantzig'],
                 df_results['gauss_L2_norm_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("Gauss dict L2 norm")

    ##### lapl_gauss_not_dict
    ###############
    #KL div
    plt.subplot(427)
    boxplotElements = plt.boxplot([df_results['lapl_gauss_not_dict_KL_div_kde'],
                 df_results['lapl_gauss_not_dict_KL_div_kde_hsj'],
                 df_results['lapl_gauss_not_dict_KL_div_adapative_dantzig'],
                 df_results['lapl_gauss_not_dict_KL_div_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("lapl_gauss_not_dict dict KL div")
    #L2 norm
    plt.subplot(428)
    boxplotElements = plt.boxplot([df_results['lapl_gauss_not_dict_L2_norm_kde'],
                 df_results['lapl_gauss_not_dict_L2_norm_kde_hsj'],
                 df_results['lapl_gauss_not_dict_L2_norm_adapative_dantzig'],
                 df_results['lapl_gauss_not_dict_L2_norm_weight_estim']])
    plt.gca().axes.xaxis.set_ticklabels(['KDE', 'KDE-SHJ', 'AD', 'MLE'])
    boxplot_params(boxplotElements)
    plt.title("lapl_gauss_not_dict dict L2 norm")
    
    plt.savefig("./plot_"+str(uuid.uuid4()), dpi=300)

if __name__ == "__main__":
    folder = "./simus_n_500_27-04-2017/"
    onlyfiles = get_files(folder)
    data_tree = generate_data_tree(onlyfiles, folder)
    df = generate_dataframe(data_tree, folder)
    generate_plot(df)

