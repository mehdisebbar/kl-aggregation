import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
TYPE_DENSITIES_LIST = ['uniform', 'rect', 'gauss', 'lapl_gauss','lapl_gauss_not_dict'] 

def plot_1graph_loss(df, type_dens, metric, file_complement):uhhu7uuyyuuuu
kjji    boxplot = sns.boxplot(x="method", y="Loss", hue="N", data=df[(df.metric==metric) & (df.type_dens==type_dens)],
                         palette=["#ff6666", "#ffd966", "#b3ff66"])
    for patch in boxplot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 1))
    boxplot.set_xlabel('')
    if metric == 'L2':
        boxplot.set_ylabel(r'$L_2$-Loss')
    else:
        boxplot.set_ylabel(metric+'-Loss')
    sns_plot = boxplot.get_figure()
    for ax in sns_plot.get_axes():
        ax.set_yscale('log')
    sns_plot = boxplot.get_figure()
    sns_plot.savefig("../../phd-thesis/TeX_files/res_"+type_dens+"_"+metric+"_"+file_complement, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)

def plot_weight_estim_loss_thesis(df5, file_complement):
    for type_dens_list in TYPE_DENSITIES_LIST:
        plt.subplots(figsize=(20, 5))
        sns.set_style("whitegrid")
        sns.despine()
        sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1})
        for type_dens in [type_dens_list]:
            for metric in ["KL", "L2"]:
                plot_1graph_loss(df5, type_dens, metric, file_complement)
