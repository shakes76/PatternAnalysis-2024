"""
Author: Ewan Stanich s4742842

This file contains helper functions that are used across the VQVAE model, including loading data, initialising weights,
calculating SSIM, clearing folders, and plotting results.
"""
from tqdm import tqdm
import nibabel as nib
import cv2
import os
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
        Provided function to assist in loading data from Nifti files.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


def load_data_2d(image_names, norm_image=False, categorical=False, dtype=np.float32, get_affines=False,
                 early_stop=False):
    """
        Provided function for loading data from Nifti files.
    """
    affines = []

    # get fixed size
    num = len(image_names)
    first_case = nib.load(image_names[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(image_names)):
        nifit_image = nib.load(inName)
        in_image = nifit_image.get_fdata(caching='unchanged')  # read disk only
        affine = nifit_image.affine
        if len(in_image.shape) == 3:
            in_image = in_image[:, :, 0]  # sometimes extra dims in HipMRI_study data
        in_image = in_image.astype(dtype)
        if norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()
        if categorical:
            in_image = to_channels(in_image, dtype=dtype)
            images[i, :, :, :] = in_image
        else:
            in_image = cv2.resize(in_image, (128, 256), interpolation=cv2.INTER_LINEAR)
            images[i, :, :] = in_image

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if get_affines:
        return images, affines
    else:
        return images


def weights_init(m):
    """
        Given a model, this function initializes the weights of the model using Xavier Uniform initialization.

        Input:
            m : the model
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", class_name)


def calc_ssim(x, y):
    """
        Given two tensors, x and y, this function calculates the Structural Similarity Index (SSIM) between the two.

        Input:
            x : the first tensor
            y : the second tensor
        Output:
            mean_ssim : the average of SSIM values of the image pairing between x and y
    """
    # Ensure that the tensors are detached and moved to the CPU
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()   

    # Iterate over the batch to get the SSIM of each image pairing
    batch_size = x.shape[0]
    ssim_values = []
    for i in range(batch_size):
        ssim_val = ssim(x[i, 0], y[i, 0], data_range=1)  
        ssim_values.append(ssim_val)

    mean_ssim = np.mean(ssim_values)  # Find mean SSIM across the batch
    return mean_ssim


def clear_folder(folder_path):
    """
        Given a folder path, this function clears the folder of all files.

        Input:
            folder_path : the path to the folder to be cleared
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def folder_check(output_loc=None, model_loc=None, image_loc=None):
    """
        This function checks if the folders for the models and epoch reconstructions exist and if they do not, creates
    """
    if output_loc is not None:
        print("OUTPUT")
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
    
    if model_loc is not None:
        print("MODEL")
        if not os.path.exists(model_loc):
            os.makedirs(model_loc)
        else:
            clear_folder(model_loc)
        
    if image_loc is not None:
        print("IMAGE")
        if not os.path.exists(image_loc):
            os.makedirs(image_loc)
        else:
            clear_folder('epoch_reconstructions')

def plot_results(train_losses, ssim_scores, n_epochs, output_loc):
    """
        Plots the metrics for the model and saves the plots in the outputs folder. The metrics are the SSIM scores and
        the training loss over the epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(output_loc + 'training_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), ssim_scores, label='SSIM Scores')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Score')
    plt.title('SSIM Scores over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(output_loc + 'ssim_scores.png')
    plt.close()