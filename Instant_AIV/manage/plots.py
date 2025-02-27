# Libraries
import numpy as np
import matplotlib.pyplot as plt
import math as m
import math
import numpy as np
from typing import List, Dict
import matplotlib.tri as tri
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.decomposition import PCA
from matplotlib import cm
# Plots
def plot(points,show=False,mark=1):
  input=points
  fig = plt.figure()
  ax_x = fig.add_subplot(111, projection='3d')
  x_ = input
  l_min=np.min(x_.flatten())
  l_max=np.max(x_.flatten())
  ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2],s=mark, marker='o')
  ax_x.set_xlim([l_min,l_max])
  ax_x.set_ylim([l_min,l_max])
  ax_x.set_zlim([l_min,l_max])
  plt.axis('off')
  if show==True:
    ax_x.view_init(azim=1, elev=90)
    plt.axis('off')
  
def extract_losses_grid(log_loss):
    
    titles = list(log_loss[0].keys())
    
    # Make sure the subplot grid dimensions match the number of titles
    num_rows = (len(titles) + 2) // 3  # Add 2 to round up
    
    # Ensure axs is a 2D array
    if num_rows == 1:
        axs = np.array([axs])


    plot_its=0
    time_all=0
    it_all=0
    for i, title in enumerate(titles):
        if title=='it':
            it_all = [entry[title] for entry in log_loss]
            plot_its=1
        if title=='time':
            time_all = [entry[title] for entry in log_loss]
            
    loss_dict={}
    for i, title in enumerate(titles):
        row = i // 3
        col = i % 3

        # Extract values for a specific loss from all dictionary entries
        loss_values = [entry[title] for entry in log_loss]
        loss_dict[title]=loss_values


    if plot_its:
        return it_all,time_all,loss_dict


def plot_losses_grid(log_loss,num_cols=3,fig_h=16,fig_v=12):
    
    titles = list(log_loss[0].keys())
    
    # Make sure the subplot grid dimensions match the number of titles
    num_rows = (len(titles) + 2) // 3  # Add 2 to round up
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_h, fig_v))

    # Ensure axs is a 2D array
    if num_rows == 1:
        axs = np.array([axs])


    plot_its=0
    time_all=0
    it_all=0
    for i, title in enumerate(titles):
        if title=='it':
            it_all = [entry[title] for entry in log_loss]
            plot_its=1
        if title=='time':
            time_all = [entry[title] for entry in log_loss]

    for i, title in enumerate(titles):
        row = i // 3
        col = i % 3
        ax = axs[row][col]

        # Extract values for a specific loss from all dictionary entries
        loss_values = [entry[title] for entry in log_loss]

        if plot_its:
            ax.plot(it_all,loss_values, label=title, color='k')
        else:
            ax.plot(loss_values, label=title, color='k')
        ax.set_title(title)
        ax.set_yscale('log')  # set y-axis to log scale
        ax.grid(True, which="both", ls="--", c='0.65')

    for ax in axs[-1, :]:
        ax.set_xlabel('Iterations (10e2)')

    plt.tight_layout()
    if plot_its:
        return it_all,time_all


def get_pdf(u_hist,bins=30):
    hist,bins=np.histogram(u_hist,bins=bins,density=True)
    prob_density=hist
    return prob_density,bins
def get_colors_plot(cmap='RdBu',n_colors=2):
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    return colors

