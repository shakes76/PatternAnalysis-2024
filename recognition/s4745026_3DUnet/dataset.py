import numpy as np
import nibabel as nib
from tqdm import tqdm
import utils
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import torch.nn as nn


def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, get_affines=False, orient=False, early_stop=False):
    affines = []
    # Get fixed size
    num = len(image_names)

    nifti_image = nib.load(image_names[0])
    first_case = nifti_image.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = utils.to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, in_name in enumerate(tqdm(image_names)):
        nifti_image = nib.load(in_name)
        inImage = nifti_image.get_fdata(caching='unchanged')  # read disk only
        affine = nifti_image.affine
        if len(inImage.shape) == 4:
            # sometimes extra dims in HipMRI_study data
            inImage = inImage[:, :, :, 0]
        inImage = inImage.astype(dtype)

        if norm_image:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            # inImage = utils.to_channels(inImage, dtype=dtype)
            inImage = utils.to_channels(inImage, dtype=dtype)
            # ~ images [i ,: ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1],
                   :inImage.shape[2], :] = inImage  # with pad

        else:
            # ~ images [i ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1],
                   :inImage.shape[2]] = inImage  # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if get_affines:
        return images, affines
    else:
        return images


class Custom3DDataset(Dataset):
    def __init__(self):
        self.image_paths = glob.glob(
            f'{"/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"}/**/*.nii.gz', recursive=True)
        self.label_paths = glob.glob(
            f'{"/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"}/**/*.nii.gz', recursive=True)

        self.upsample = torch.nn.Upsample(
            size=(128, 128, 128), mode="trilinear")
        # self.classes = {i: i for i in range(6)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_data_3D([self.image_paths[idx]])
        label = load_data_3D([self.label_paths[idx]])

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        label = nn.functional.one_hot(
            label, num_classes=6).permute(3, 1, 0, 2).float()
        image = self.upsample(image)
        label = self.upsample(label)

        return image.squeeze(0), label
