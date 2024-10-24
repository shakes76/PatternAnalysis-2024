""" 
File: dataset.py
Author: Øystein Kvandal
Description: Contains the functions for loading the medical image data from the dataset for the UNET model.
"""

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torchio as tio
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# Dataset path
# root_path = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/HipMRI_study_keras_slices_data/keras_slices_' # Local path
root_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_' # Rangpur path


def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr. shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c: c +1][ arr == c ] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre - allocates 4D arrays for conv2d to avoid excessive memory &
    usage.

    normImage: bool (normalise the image 0.0 -1.0)
    early_stop: Stop loading pre - maturely, leaves arrays mostly empty, for quick &
    loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case [:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        print(f'Image shape: {rows}x{cols}x{channels}')
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return torch.tensor(images, dtype = torch.float32), affines
    else:
        return torch.tensor(images, dtype = torch.float32)


# def load_img_seg_pair(dataset_type="train"):
#     assert dataset_type in ["train", "test", "validate"], "Invalid dataset type. Must be 'train', 'test' or 'validate'."

#     img_path = root_path + dataset_type + '/'
#     seg_path = root_path + 'seg_' + dataset_type + '/'
#     images_paths = sorted([os.path.join(img_path, img) for img in os.listdir(img_path) if img.endswith('.nii.gz')])
#     segmentations_paths =  sorted([os.path.join(seg_path, seg) for seg in os.listdir(seg_path) if seg.endswith('.nii.gz')])
#     images = load_data_2D(images_paths[342:350], normImage=True)
#     segmentations = load_data_2D(segmentations_paths[342:350])

#     return images, segmentations


class MRIDataset(data.Dataset):
    def __init__(self, dataset_type):
        self.image_paths = sorted([os.path.join(root_path + dataset_type + '/', img) for img in os.listdir(root_path + dataset_type + '/') if img.endswith('.nii.gz')])
        self.label_paths = sorted([os.path.join(root_path + 'seg_' + dataset_type + '/', seg) for seg in os.listdir(root_path + 'seg_' + dataset_type + '/') if seg.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_data_2D([self.image_paths[idx]], normImage=True)
        label = load_data_2D([self.label_paths[idx]])
        image = transforms.Resize((64, 64))(image)
        label = transforms.Resize((64, 64))(label)

        return image, label


class MRIDataLoader(data.DataLoader):
    def __init__(self, dataset_type, batch_size=1, shuffle=True):
        self.dataset = MRIDataset(dataset_type)
        super(MRIDataLoader, self).__init__(self.dataset, batch_size=batch_size, shuffle=shuffle)



### Unit test

# from matplotlib import pyplot as plt

# test_dataset = MRIDataset('train')

# plt.figure()
# cols, rows = 2, 2
# for i in range(rows*cols):
#     img, seg = test_dataset[i]
#     plt.subplot(rows, cols, i+1)
#     plt.imshow(img[0], cmap='gray')
#     plt.imshow(seg[0], cmap='gray', alpha=0.5)

# plt.show()


# for i, image in enumerate(images):
#     plt.figure(i)
#     plt.subplot(1, 7, 1)
#     plt.imshow(torch.eq(image, 0), cmap='gray')
#     plt.subplot(1, 7, 2)
#     plt.imshow(torch.eq(image, 1), cmap='gray')
#     plt.subplot(1, 7, 3)
#     plt.imshow(torch.eq(image, 2), cmap='gray')
#     plt.subplot(1, 7, 4)
#     plt.imshow(torch.eq(image, 3), cmap='gray')
#     plt.subplot(1, 7, 5)
#     plt.imshow(torch.eq(image, 4), cmap='gray')
#     plt.subplot(1, 7, 6)
#     plt.imshow(torch.eq(image, 5), cmap='gray')
#     plt.subplot(1, 7, 7)
#     plt.imshow(image, cmap='gray')
#     plt.show()
