"""A file with helper methods like plotting"""
import matplotlib.pyplot as plt
import numpy as np

SEED = 31

def save_plot(train, validation, num_epochs, save_to, title, y_limits):
    
    x = [i+1 for i in range(num_epochs)]
    x_ticks = np.arange(0, num_epochs+1, 5)
    x_ticks[0] = 1
    plt.plot(x, train, label="Train")
    plt.plot(x, validation, label="Validation")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.ylim(y_limits)
    plt.xlim((1, num_epochs))
    plt.xticks(x_ticks)

    plt.savefig(save_to)
    plt.clf()

def plot_umap():
    pass