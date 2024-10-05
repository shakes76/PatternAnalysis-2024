import matplotlib.pyplot as plt
import numpy as np

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
# display a random image from the dataset
def display_random_images(images, labels=None, n=1):
    indices = np.random.choice(len(images), n)
    images = images[indices]
    if labels is not None:
        labels = labels[indices]
    image_shape = images[0].shape
    plot_gallery(images, labels, image_shape[0], image_shape[1])
    plt.show()