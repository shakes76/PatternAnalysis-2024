import numpy as np
import nibabel as nib
from tqdm import tqdm
import utils
import torch
from torch.utils.data import Dataset


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


class Prostate3DDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata(dtype=np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata(dtype=np.uint8)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
            image, label = sample["image"], sample["label"]

        return image, label
