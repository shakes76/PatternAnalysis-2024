"""A file with helper methods like plotting"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

SEED = 31

# plot the loss and accuracy for both train and validation
def save_plot(train, validation, num_epochs, save_to, title, y_limits):
    
    x = [i+1 for i in range(num_epochs)]
    x_ticks = np.arange(0, num_epochs+1, 50)
    x_ticks[0] = 1
    plt.figure(figsize=(10, 10))
    plt.plot(x, train, label="Train")
    plt.plot(x, validation, label="Validation")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.ylim(y_limits)
    plt.xlim((1, num_epochs))
    plt.xticks(x_ticks)

    plt.savefig(save_to)
    plt.clf()

# Plot the t-SNE of the model predictions predicted vs true
def plot_tsne(out, y_true):
    tsne = TSNE()
    x_new = tsne.fit_transform(out)


    x = []
    y = []
    for point in x_new:
        x.append(point[0])
        y.append(point[1])
    
    plt.figure(figsize=(10, 10))
    for i in range(4):
        x = []
        y = []
        for point in x_new[y_true == i]:
            x.append(point[0])
            y.append(point[1])
        
        plt.scatter(x, y, label=f"{i}")
    plt.title("t-SNE Predicted vs actual")
    plt.savefig("./images/scatter.png")
    plt.clf()
