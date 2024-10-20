""" Loads 2D Nifti image files into Pytorch Dataset

"""

import os
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ProstateDataset(Dataset):
    """ 
    Dataset used to store the Prostate image files.
    
    Modified from Pytorch tutorial:
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 
    
    """

    def __init__(self, img_dir: str, mask_dir: str, early_stop = True, plotting = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = utils.load_data_2D_from_directory(self.img_dir, early_stop=early_stop)
        self.masks = utils.load_data_2D_from_directory(self.mask_dir, early_stop=early_stop, categorical=True)
        if plotting:
            self.masks_for_plotting = utils.load_data_2D_from_directory(self.mask_dir, early_stop=early_stop, categorical=False)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        # Move channels to 0th index and convert np to tensor
        mask = np.moveaxis(mask, 2, 0)
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
    
        return (img, mask)
    
    def img_show(self, start_idx: int = 0):
        """ Plots 6 of the image files, and their corresponding masks.
        
        Modified from Pytorch tutorial: 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        """

        fig = plt.figure(figsize=(8, 8))
        cols, rows = 4, 3
        count = 0
        for idx in range(start_idx, start_idx + (cols * rows) // 2):
            img = self.imgs[idx]
            mask = self.masks_for_plotting[idx]
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis("off")
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(mask.squeeze(), cmap='gray')
            plt.axis("off")
        plt.show()

# For testing purposes
if __name__ == "__main__":
    main_dir = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data')
    train_image_path = os.path.join(main_dir, 'keras_slices_train')
    train_mask_path = os.path.join(main_dir, 'keras_slices_seg_train')
    test_image_path = os.path.join(main_dir, 'keras_slices_test')
    test_mask_path = os.path.join(main_dir, 'keras_slices_seg_test')

    train_dataset = ProstateDataset(train_image_path, train_mask_path, plotting=True)
    print(len(train_dataset))
    print(type(train_dataset[1][1]))
    print(train_dataset[1][0].shape)
    print(train_dataset[1][1].shape)
    train_dataset.img_show(15)