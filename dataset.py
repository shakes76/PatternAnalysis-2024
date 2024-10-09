
import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1

    return res

# Placeholder for the orientation correction function
def applyOrientation(nifti_image, interpolation='linear', scale=1):
    # Implement or import orientation correction logic here
    return nifti_image  # Replace with actual implementation

class NiftiDataset(Dataset):
    def __init__(
        self,
        image_filenames,
        label_filenames,
        normImage=False,
        categorical=False,
        dtype=np.float32,
        getAffines=False,
        orient=False,
        transform=None,
    ):
        self.image_filenames = list(image_filenames)
        self.label_filenames = list(label_filenames)
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.getAffines = getAffines
        self.orient = orient
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        inName = self.image_filenames[idx]
        niftiImage = nib.load(inName)
        interp = "nearest" if self.dtype == np.uint8 else "linear"

        if self.orient:
            niftiImage = applyOrientation(niftiImage, interpolation=interp, scale=1)

        inImage = niftiImage.get_fdata(caching="unchanged")

        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # Remove extra dimension if present

        inImage = inImage.astype(self.dtype)

        if self.normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if self.categorical:
            inImage = to_channels(inImage, dtype=self.dtype)

        if self.transform:
            inImage = self.transform(inImage)

        inImage = torch.from_numpy(inImage)

        # **Add channels dimension to image**
        inImage = inImage.unsqueeze(0)  # Shape becomes (1, D, H, W)

        # Load label
        labelName = self.label_filenames[idx]
        labelImage = nib.load(labelName).get_fdata(caching="unchanged")

        if len(labelImage.shape) == 4:
            labelImage = labelImage[:, :, :, 0]  # Remove extra dimension if present

        # labelImage = labelImage.astype(self.dtype)
        labelImage = labelImage.astype(np.int64)

        if self.categorical:
            labelImage = to_channels(labelImage, dtype=self.dtype)

        if self.transform:
            labelImage = self.transform(labelImage)

        labelImage = torch.from_numpy(labelImage)

        # **Add channels dimension to label**
        # labelImage = torch.from_numpy(to_channels(labelImage, dtype=self.dtype))

        labelImage = labelImage.unsqueeze(0)  # Shape becomes (1, D, H, W)

        # print(f"inImage shape after adding channels dimension: {inImage.shape}")
        # print(f"labelImage shape after adding channels dimension: {labelImage.shape}")
        # inImage shape after adding channels dimension: torch.Size([1, 256, 256, 128])
        # labelImage shape after adding channels dimension: torch.Size([1, 256, 256, 128])



        if self.getAffines:
            return inImage, labelImage, affine
        else:
            return inImage, labelImage



# def load_data_3D(
#     imageNames,
#     normImage=False,
#     categorical=False,
#     dtype=np.float32,
#     getAffines=False,
#     orient=False,
#     early_stop=False,
# ):
#     """
#     Load medical image data from names, cases list provided into a list for each.
#     This function pre - allocates 5D arrays for conv3d to avoid excessive memory &
#     usage .
#     normImage : bool ( normalise the image 0.0 -1.0)
#     orient : Apply orientation and resample image ? Good for images with large slice &
#     thickness or anisotropic resolution
#     dtype : Type of the data . If dtype =np.uint8 , it is assumed that the data is &
#     labels
#     early_stop : Stop loading pre - maturely ? Leaves arrays mostly empty , for quick &
#     loading and testing scripts .
#     """
#     affines = []

#     # ~ interp = ' continuous '
#     interp = "linear "
#     if dtype == np.uint8:  # assume labels
#         interp = "nearest "

#     # get fixed size
#     num = len(imageNames)
#     niftiImage = nib.load(imageNames[0])
#     if orient:
#         niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
#         # ~ testResultName = "oriented.nii.gz"
#         # ~ niftiImage.to_filename(testResultName)
#     first_case = niftiImage.get_fdata(caching="unchanged")
#     if len(first_case.shape) == 4:
#         first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
#     if categorical:
#         first_case = to_channels(first_case, dtype=dtype)
#         rows, cols, depth, channels = first_case.shape
#         images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
#     else:
#         rows, cols, depth = first_case.shape
#         images = np.zeros((num, rows, cols, depth), dtype=dtype)

#     for i, inName in enumerate(tqdm(imageNames)):
#         niftiImage = nib.load(inName)
#         if orient:
#             niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
#         inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
#         affine = niftiImage.affine
#         if len(inImage.shape) == 4:
#             inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
#         inImage = inImage[:, :, :depth]  # clip slices
#         inImage = inImage.astype(dtype)
#         if normImage:
#             # ~ inImage = inImage / np.linalg.norm(inImage)
#             # ~ inImage = 255. * inImage / inImage.max()
#             inImage = (inImage - inImage.mean()) / inImage.std()
#         if categorical:
#             inImage = to_channels(inImage, dtype=dtype)
#             # ~ images[i, :, :, :, :] = inImage
#             images[
#                 i,
#                 : inImage.shape[0],
#                 : inImage.shape[1],
#                 : inImage.shape[2],
#                 : inImage.shape[3],
#             ] = inImage  # with pad
#         else:
#             # ~ images[i, :, :, :] = inImage
#             images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]] = (
#                 inImage  # with pad
#             )

#         affines.append(affine)
#         if i > 20 and early_stop:
#             break

#     if getAffines:
#         return images, affines
#     else:
#         return images

