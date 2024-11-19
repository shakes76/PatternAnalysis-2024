"""
The helper methods used throughout.

Made by Joshua Deadman
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

from config import IMAGEPATH, TESTING, VALIDATION

def split_data(label_path) -> tuple[list, list, list]:
    """ Splits the labels into a training, testing and validation set.
        The data is split with a 1:1 amount of benign and malignant data, omitting excess benign images.

    Arguments:
        label_path (str): The absolute path to .csv file holding labels.
    Returns:
        A tuple of lists holding dicts with the isic_id and target value.
        The order of lists is (Training, Testing, Validaiton).
    """
    df = pd.read_csv(label_path)
    malignant_count = sum(df["target"] == 1)
    malignant = df[df["target"] == 1]
    malignant = malignant.drop("patient_id", axis="columns")
    # Reduce benign samples to the same size as malignant
    benign = df.drop(labels=malignant.index.to_list()).sample(n=malignant_count)
    benign = benign.drop("patient_id", axis="columns")

    # Exclusive sets of malignant images
    test_m = malignant.sample(n=math.floor(malignant_count * (TESTING+VALIDATION)))
    val_m = test_m.sample(n=math.floor(malignant_count * VALIDATION))
    train_m = malignant.drop(labels=test_m.index.to_list())
    test_m = test_m.drop(labels=val_m.index.to_list())
    
    # Exclusive sets of benign data
    test_b = benign.sample(n=math.floor(malignant_count * (TESTING+VALIDATION)))
    val_b = test_b.sample(n=math.floor(malignant_count * VALIDATION))
    train_b = benign.drop(labels=test_b.index.to_list())
    test_b = test_b.drop(labels=val_b.index.to_list())

    # Sample used to randomise the order
    train = pd.concat([train_m, train_b]).sample(frac=1)
    test = pd.concat([test_m, test_b]).sample(frac=1)
    val = pd.concat([val_m, val_b]).sample(frac=1)

    return train.to_dict(orient="records"), test.to_dict(orient="records"), val.to_dict(orient="records")

def generate_loss_plot(training_loss, validation_loss, model, save=False) -> None:
    """ Plots the training loss against the validation_loss.

    Arguments:
        training_loss (list): The average loss per epoch while training.
        validation_loss (list): The average loss per epoch while validating.
        model (str): Should be the name of the model relevant to the losses.
            This string should not have any / or other characters relevant to paths.
        save (bool): True - Save image to path defined in config.py.
                     False - Show plot.
    """
    plt.figure()
    plt.title("Loss of the " + model)
    plt.plot(training_loss, label="Training")
    plt.plot(validation_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if save:
        plt.savefig(os.path.join(IMAGEPATH, "loss_" + model.lower().replace(" ","") + ".png"))
    else:
        plt.show()

def tsne_plot(features, labels, save=False) -> None:
    """ Plots the t-SNE embeddings to show feature extraction.
        This method was reworked from the following source:
        https://github.com/2-Chae/PyTorch-tSNE/blob/main/main.py

    Arguments:
        features (list): Features extracted from the siamese network. Should be on cpu already.
        labels (list): labels respective to the features. Should be on cpu already.
        save (bool): True - Save image to path defined in config.py.
                     False - Show plot.
    """
    plt.figure()
    features = np.concatenate(features, axis=0).astype(np.float64)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE()
    tsne_output = tsne.fit_transform(features)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['Targets'] = labels
    df = df.replace({0: "Benign", 1: "Malignant"})

    plt.rcParams['figure.figsize'] = 15, 20
    sns.scatterplot(
        x='x', y='y',
        hue='Targets',
        palette=sns.color_palette("hls", 2),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
    if save:
        plt.savefig(os.path.join(IMAGEPATH, "tsne_scatterplot.png"))
    else:
        plt.show()
