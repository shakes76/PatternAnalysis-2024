
# import os
# import glob
# import torch
# import numpy as np
# import nibabel as nib
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm


# def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
#     channels = np.unique(arr)
#     res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
#     for c in channels:
#         c = int(c)
#         res[..., c : c + 1][arr == c] = 1

#     return res

# # Placeholder for the orientation correction function
# def applyOrientation(nifti_image, interpolation='linear', scale=1):
#     # Implement or import orientation correction logic here
#     return nifti_image  # Replace with actual implementation

# class NiftiDataset(Dataset):
#     def __init__(
#         self,
#         image_filenames,
#         label_filenames,
#         normImage=False,
#         categorical=False,
#         dtype=np.float32,
#         getAffines=False,
#         orient=False,
#         transform=None,
#     ):
#         self.image_filenames = list(image_filenames)
#         self.label_filenames = list(label_filenames)
#         self.normImage = normImage
#         self.categorical = categorical
#         self.dtype = dtype
#         self.getAffines = getAffines
#         self.orient = orient
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         # Load image
#         inName = self.image_filenames[idx]
#         niftiImage = nib.load(inName)
#         interp = "nearest" if self.dtype == np.uint8 else "linear"

#         if self.orient:
#             niftiImage = applyOrientation(niftiImage, interpolation=interp, scale=1)

#         inImage = niftiImage.get_fdata(caching="unchanged")

#         affine = niftiImage.affine

#         if len(inImage.shape) == 4:
#             inImage = inImage[:, :, :, 0]  # Remove extra dimension if present

#         inImage = inImage.astype(self.dtype)

#         if self.normImage:
#             inImage = (inImage - inImage.mean()) / inImage.std()

#         if self.categorical:
#             inImage = to_channels(inImage, dtype=self.dtype)

#         if self.transform:
#             inImage = self.transform(inImage)

#         inImage = torch.from_numpy(inImage)

#         # **Add channels dimension to image**
#         inImage = inImage.unsqueeze(0)  # Shape becomes (1, D, H, W)

#         # Load label
#         labelName = self.label_filenames[idx]
#         labelImage = nib.load(labelName).get_fdata(caching="unchanged")

#         if len(labelImage.shape) == 4:
#             labelImage = labelImage[:, :, :, 0]  # Remove extra dimension if present

#         # labelImage = labelImage.astype(self.dtype)
#         labelImage = labelImage.astype(np.int64)

#         if self.categorical:
#             labelImage = to_channels(labelImage, dtype=self.dtype)

#         if self.transform:
#             labelImage = self.transform(labelImage)

#         labelImage = torch.from_numpy(labelImage)
#         labelImage = labelImage.unsqueeze(0)  # Shape becomes (1, D, H, W)

#         if self.getAffines:
#             return inImage, labelImage, affine
#         else:
#             return inImage, labelImage


import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NiftiDataset(Dataset):
    def __init__(
        self,
        image_filenames,
        label_filenames,
        transform=None,
        dtype=np.float32,
    ):
        self.image_filenames = list(image_filenames)
        self.label_filenames = list(label_filenames)
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        inName = self.image_filenames[idx]
        niftiImage = nib.load(inName)

        inImage = niftiImage.get_fdata(caching="unchanged")

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # Remove extra dimension if present

        inImage = inImage.astype(self.dtype)

        # if self.normImage:
        #     inImage = (inImage - inImage.mean()) / inImage.std()

        inImage = torch.from_numpy(inImage)
        # **Add channels dimension to image**
        inImage = inImage.unsqueeze(0)  # Shape becomes (1, D, H, W)

        # Load label
        labelName = self.label_filenames[idx]
        labelImage = nib.load(labelName).get_fdata(caching="unchanged")

        if len(labelImage.shape) == 4:
            labelImage = labelImage[:, :, :, 0]  # Remove extra dimension if present

        labelImage = labelImage.astype(np.int64)

        labelImage = torch.from_numpy(labelImage)
        labelImage = labelImage.unsqueeze(0)  # Shape becomes (1, D, H, W)
        
        if self.transform:
            inImage = self.transform(inImage)
            labelImage = self.transform(labelImage)
        
        return inImage, labelImage

