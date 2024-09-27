import torch
import os

import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib

from torchvision import transforms
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1

    return res


def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    orient=False,
    early_stop=False,
):
    """
    Load medical image data from names , cases list provided into a list for each .
    This function pre - allocates 5D arrays for conv3d to avoid excessive memory &
    usage .
    normImage : bool ( normalise the image 0.0 -1.0)
    orient : Apply orientation and resample image ? Good for images with large slice &
    thickness or anisotropic resolution
    dtype : Type of the data . If dtype =np.uint8 , it is assumed that the data is &
    labels
    1early_stop : Stop loading pre - maturely ? Leaves arrays mostly empty , for quick &
    loading and testing scripts .
    """
    affines = []

    # ~ interp = ' continuous '
    interp = "linear "
    if dtype == np.uint8:  # assume labels
        interp = "nearest "

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        # ~ testResultName = "oriented.nii.gz"
        # ~ niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = utils.to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            # ~ images[i, :, :, :, :] = inImage
            images[
                i,
                : inImage.shape[0],
                : inImage.shape[1],
                : inImage.shape[2],
                : inImage.shape[3],
            ] = inImage  # with pad
        else:
            # ~ images[i, :, :, :] = inImage
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]] = (
                inImage  # with pad
            )

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


slice_data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
semantic_labels_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
semantic_MR_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"


class NormConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation=nn.ReLU(),
    ):
        super(NormConv3D, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm = nn.BatchNorm3d(out_channels)

        self.activation = activation

    def foward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class NormConvTranspose3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation=nn.ReLU(),
    ):
        super(NormConv3D, self).__init__()

        # Convolutional layer
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm = nn.BatchNorm3d(out_channels)

        self.activation = activation

    def foward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self.__init__)

        # Encoder
        self.enc11 = NormConv3D(3, 32, kernel_size=3, stride=1, padding=1)
        self.enc12 = NormConv3D(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc21 = NormConv3D(64, 64, kernel_size=3, stride=1, padding=1)
        self.enc22 = NormConv3D(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc31 = NormConv3D(128, 128, kernel_size=3, stride=1, padding=1)
        self.enc32 = NormConv3D(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.botteneck1 = NormConv3D(256, 256, kernel_size=3, stride=1, padding=1)
        self.botteneck2 = NormConv3D(256, 512, kernel_size=3, stride=1, padding=1)


        # Decoder
        self.up1 = NormConvTranspose3D(256, 256, kernel_size=2, stride=2)
        self.dec11 = NormConv3D(768, 256, kernel_size=3, stride=1, padding=1)
        self.dec12 = NormConv3D(256, 256, kernel_size=3, stride=1, padding=1)

        self.up2 = NormConvTranspose3D(256, 256, kernel_size=2, stride=2)
        self.dec21 = NormConv3D(384, 128, kernel_size=3, stride=1, padding=1)
        self.dec22 = NormConv3D(128, 128, kernel_size=3, stride=1, padding=1)

        self.up3 = NormConvTranspose3D(128, 128, kernel_size=2, stride=2)
        self.dec31 = NormConv3D(192, 64, kernel_size=3, stride=1, padding=1)
        self.dec32 = NormConv3D(64, 64, kernel_size=3, stride=1, padding=1)

        self.final = nn.Conv3d(64, 3, kernel_size=1) # TODO check if this is right

    def foward(self, x):
        # Encocder foward pass
        e11 = self.enc11(x)
        e12 = self.enc12(e11)
        p1 = self.pool1(e12)

        e21 = self.enc21(p1)
        e22 = self.enc22(e21)
        p2 = self.pool2(e22)

        e31 = self.enc31(p2)
        e32 = self.enc32(e31)
        p3 = self.pool3(e32)

        # Bottleneck
        b1 = self.botteneck1(p3)
        b2 = self.botteneck2(b1)

        #Decoder foward pass with skip connections
        d1 = self.up1(b2)
        d1 = torch.cat([e32, d1], dim=1) # TODO check if dim is correct
        d1 = self.dec11(d1)
        d1 = self.dec12(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([e22, d2], dim=1)
        d2 = self.dec21(d2)
        d2 = self.dec22(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([e12, d3], dim=1)
        d3 = self.dec31(d3)
        d3 = self.dec32(d3)

        out = self.final(d3)
        return out


