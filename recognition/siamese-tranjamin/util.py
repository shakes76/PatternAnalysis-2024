from dataset import BalancedMelanomaDataset
import numpy as np
import matplotlib.pyplot as plt


def plot_images(square_size=3):
    '''
    Plots a selection of preprocessed images.
    '''
    df = BalancedMelanomaDataset(
        image_shape=(256, 256),
        batch_size=64,
        validation_split=0.2,
        balance_split=0.5
    )

    for batch in df.dataset:
        features, labels = batch

        plt.figure(figsize=(1.8 * square_size, 2.4 * square_size))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(square_size * square_size):
            plt.subplot(square_size, square_size, i + 1)
            plt.imshow(np.array(features[i]))
            plt.title("Benign" if int(labels[i]) == 0 else "Malignant", size=12)
            plt.xticks(())
            plt.yticks(())

        break

plot_images(square_size=8)
plt.show()