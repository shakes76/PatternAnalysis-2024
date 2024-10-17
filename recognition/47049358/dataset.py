# ==========================
# Imports
# ==========================
import os
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
import torch

# ==========================
# Constants
# ==========================

IMAGE_FILE_NAME = os.path.join(os.getcwd(), 'semantic_MRs_anon')
LABEL_FILE_NAME = os.path.join(os.getcwd(), 'semantic_labels_anon')

# ==========================
# Loading Dataset
# ==========================

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        if categorical:
            # inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break

    return (images, affines) if getAffines else images

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    dtype : Type of the data. If dtype=np.uint8, it is assumed that the data has labels.
    early_stop : Stop loading prematurely. Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    first_case = niftiImage.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        
        niftiImage = nib.load(inName)
        
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 4:
            inImage = inImage[ : , : , : , 0]  # sometimes extra dims in HipMRI_study data
        
        inImage = inImage[ : , : , : depth]  # clip slices
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2], : inImage.shape[3]] = inImage  # with pad
        else:
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]] = inImage  # with pad

        affines.append(affine)

        if i > 20 and early_stop:
            break

    return images

def load_images_and_labels(image_file_name, label_file_name, early_stop = False):
    images = load_data_3D(imageNames=image_file_name, normImage = True, early_stop=early_stop)
    labels = load_data_3D(imageNames=label_file_name, categorical=True, dtype = np.uint8, early_stop=early_stop)
    return images, labels

class Prostate3dDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]
        return img, mask
    
# ==========================
# Image Augmentation
# ==========================

def random_rotation(image: torch.Tensor, angle = 0.0):
        
    image = F.rotate(image, angle = angle, fill = image.min().item())

    return image

def random_resize_with_padding(image: torch.Tensor, height = 256, width = 256, is_mask = False):

    original_height = image.size(1)
    original_width = image.size(2)

    top_p = (original_height - height) // 2
    bottom_p = (original_height - height) - top_p
    right_p = (original_width - width) // 2
    left_p = (original_width - width) - right_p
    resized = F.resize(image, size = (height, width))
    padding = (left_p, top_p, right_p, bottom_p)
    image = F.pad(resized, padding = padding, fill = image.min().item())

    if is_mask:
        image = torch.clamp(torch.round(image), min = 0, max = 1)

    return image

def gamma_correction(image: torch.Tensor, gamma = 1):
        
    image = F.adjust_gamma(image, gamma)
    image = torch.nan_to_num(image, nan = 0)

    return image

def elastic_transformation(image: torch.Tensor, alpha = 5.0, sigma = 0.5, is_mask = False):

    elastic_transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ElasticTransform(alpha=alpha, sigma=sigma),
            transforms.PILToTensor()
    ])

    if is_mask:

        mask = elastic_transformation(image)
        mask = torch.clamp(torch.round(mask), min = 0, max = 1)

        return mask

    image = elastic_transformation(image)

    return image

def apply_transformation(image: torch.Tensor, masks_3d: torch.Tensor):

    alpha = random.uniform(0.0, 10.0)
    sigma = random.uniform(0.0, 1.0)
    make_elastic = random.choice([True, False])
    
    angle = random.uniform(0, 180)
    rotate = random.choice([True, False])

    height = random.randint(128, 256)
    width = random.randint(128, 256)
    resize = random.choice([True, False])
        
    gamma = random.uniform(0.25, 2)
    adjust_gamma = random.choice([True, False])
    
    hflip = random.choice([True, False])

    vflip = random.choice([True, False])

    for j in range(image.shape[-1]):

        slice = image[ : , : , j]
        slice = torch.Tensor(slice[np.newaxis, : , :])
        masks = masks_3d[: , : , j , :]
        masks = torch.Tensor(masks[np.newaxis, : , :, : ])

        slice = elastic_transformation(image = slice, alpha = alpha, sigma = sigma) if make_elastic else slice
        slice = random_rotation(image = slice, angle = angle) if rotate else slice
        slice = random_resize_with_padding(image = slice, height = height, width = width) if resize else slice
        slice = gamma_correction(image = slice, gamma = gamma) if adjust_gamma else slice
        slice = F.hflip(slice) if hflip else slice
        slice = F.vflip(slice) if vflip else slice

        for i in range(masks.size(-1)):
            mask = masks[ : , : , : , i]

            mask = elastic_transformation(image = mask, alpha = alpha, sigma = sigma, is_mask = True) if make_elastic else mask
            mask = random_rotation(image = mask, angle = angle) if rotate else mask
            mask = random_resize_with_padding(image = mask, height = height, width = width, is_mask=True) if resize else mask
            mask = F.hflip(mask) if hflip else mask
            mask = F.vflip(mask) if vflip else mask

            masks[ : , : , : , i] = mask

        masks_3d[: , : , j , :] = masks.cpu().numpy()
        
        image[ : , : , j] = slice.cpu().numpy()
        
    return image, masks_3d

def augment_training_set(X_train, y_train):

    for i in range(X_train.shape[0]):

        image = X_train[i, : , : , : ]
        masks_3d = y_train[i, : , : , : , :]

        transformed_img, transformed_masks_3d = apply_transformation(image = image, masks_3d = masks_3d)

        X_train[i, : , : , : ] = transformed_img
        y_train[i, : , : , : , :] = transformed_masks_3d

    return X_train, y_train

print('> Loading Dataset')

rawImageNames = os.listdir(IMAGE_FILE_NAME)
rawLabelNames = os.listdir(LABEL_FILE_NAME)

# Split the set into train, validation, and test set (80 : 20 for train:test)
X_train, X_test, y_train, y_test = train_test_split(rawImageNames, rawLabelNames, train_size=0.8) # Split the data in training and test set

X_train = [os.path.join(IMAGE_FILE_NAME, image) for image in X_train]
X_test = [os.path.join(IMAGE_FILE_NAME, image) for image in X_test]

y_train = [os.path.join(LABEL_FILE_NAME, label) for label in y_train]
y_test = [os.path.join(LABEL_FILE_NAME, label) for label in y_test]

X_train, y_train = load_images_and_labels(X_train, y_train, early_stop=False)
X_test, y_test = load_images_and_labels(X_test, y_test, early_stop=False)

print('> Start Random Augmentation')

X_train, y_train = augment_training_set(X_train, y_train)

print('> Augmentation Complete')

X_train = X_train[: ,np.newaxis, :, :, :]
X_test = X_test[:, np.newaxis, :, :, :]

# if __name__ == '__main__':
#     # Generate random noise with shape (256, 256, 128)
#     image_file_name = [os.path.join(IMAGE_FILE_NAME, image) for image in os.listdir(IMAGE_FILE_NAME)][0]
#     noise = torch.Tensor(load_data_3D(imageNames=[image_file_name], normImage = True, early_stop=True))

#     import matplotlib.pyplot as plt

#     torch.set_printoptions(threshold=float('inf'))

#     image = noise[: , :, :, 10]

#     print(image)

#     plt.imshow(image[0, :, :], cmap='gray')
#     plt.show()
#     print(image.shape)
#     image = F.hflip(image)
#     plt.imshow(image[0, :, :], cmap='gray')
#     plt.show()

#     print(image)