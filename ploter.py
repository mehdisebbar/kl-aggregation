import matplotlib.pyplot as plt
import numpy as np

HIST_COLOR =  "#66ff66"

def histogram_plot(X, distrib, bin_size, max_y=2, fig_size=(10, 7)):
    """
    Plot the histogram of X and the true distribution
    """
    fig, ax = plt.subplots(1, 1, figsize = fig_size)
    plt.xlim(0, 1)
    plt.ylim(0, max_y)
    x = np.linspace(0, 1, distrib.shape[0])
    #We plot the true disribution
    ax.plot(x, distrib, 'r-', lw=3, alpha=0.6, label='True distribution')
    #We plot the histogram
    bins = np.linspace(0, 1, bin_size)
    ax.hist(X, bins, normed=True, histtype='bar', alpha=0.5, color=HIST_COLOR)
    ax.legend(loc='upper right', frameon=False)
    return fig

def multiple_densities_plot(densities_dict):
    pass 