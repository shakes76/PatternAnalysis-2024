""" Loads 2D Nifti image files into Pytorch Dataset

"""

import os
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class ProstateDataset(Dataset):
    """ 
    Dataset used to store the Prostate image files.
    
    Modified from Pytorch tutorial:
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 
    
    """

    def __init__(self, img_dir: str, mask_dir: str, transform = None, target_transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = load_data_2D_from_directory(self.img_dir, early_stop=True)
        self.masks = load_data_2D_from_directory(self.mask_dir, early_stop=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
    
        return img, mask
    
    def img_show(self):
        """ Plots 6 of the image files, and their corresponding masks.
        
        Modified from Pytorch tutorial: 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        """

        fig = plt.figure(figsize=(8, 8))
        cols, rows = 4, 3
        count = 0
        for idx in range(0, cols * rows):
            img = self.imgs[idx]
            mask = self.masks[idx]
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(img.squeeze(), cmap="gray")
            plt.axis("off")
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(mask.squeeze(), cmap="gray")
            plt.axis("off")
        plt.show()

def load_data_2D_from_directory(image_folder_path: str, normImage=False, categorical=False, dtype = np.float32, getAffines = False, early_stop = False) -> np.ndarray:
    """ Returns np array of all Nifti images in the specified image folder."""
    image_names = []
    for file in os.listdir(image_folder_path):
        image_names.append(os.path.join(image_folder_path, file))
    return utils.load_data_2D(image_names, normImage, categorical, dtype, getAffines, early_stop)

# For testing purposes
if __name__ == "__main__":
    main_dir = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data')
    train_image_path = os.path.join(main_dir, 'keras_slices_train')
    train_mask_path = os.path.join(main_dir, 'keras_slices_seg_train')
    test_image_path = os.path.join(main_dir, 'keras_slices_test')
    test_masks_path = os.path.join(main_dir, 'keras_slices_seg_test')

    train_dataset = ProstateDataset(train_image_path, train_mask_path)
    print(len(train_dataset))
    print(train_dataset[1])
    train_dataset.img_show()