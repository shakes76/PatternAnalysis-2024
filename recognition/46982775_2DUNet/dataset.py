"""Loads all required 2D Nifti image files from specified path using the helper functions by Shekhar "Shakes" Chandra. """

import os
import numpy as np
import utils
import torch
import torch.utils.data
import matplotlib.pyplot as plt

def load_data_2D_from_directory(image_folder_path: str, normImage=False, categorical=False, dtype = np.float32, getAffines = False, early_stop = False) -> np.ndarray:
    """ Takes the main image folder path and returns four sets of images: train and test and images and masks."""
    image_names = []
    for file in os.listdir(image_folder_path):
        image_names.append(os.path.join(image_folder_path, file))
    return utils.load_data_2D(image_names, normImage, categorical, dtype, getAffines, early_stop)

# For testing purposes
if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data\\')
    # print(os.listdir(path))
    train_image_path = os.path.join(path, 'keras_slices_train')
    train_mask_path = os.path.join(path, 'keras_slices_seg_train')
    test_image_path = os.path.join(path, 'keras_slices_test')
    test_masks_path = os.path.join(path, 'keras_slices_seg_test')

    # train_images = load_data_2D_from_directory(train_image_path)
    train_masks = load_data_2D_from_directory(train_mask_path, early_stop=True)
    # test_images = load_data_2D_from_directory(test_image_path)
    # test_masks = load_data_2D_from_directory(test_masks_path)

    plt.imshow(train_masks[1],cmap='gray')
    plt.show()
