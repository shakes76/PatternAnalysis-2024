import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

# Helper function to load NIfTI files
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

    Parameters:
    - normImage: bool (normalize the image 0.0-1.0)
    - early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    
    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Sometimes extra dims, remove
        if categorical:
            first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images
    
# Custom Dataset class for loading data into PyTorch
class HipMRIDataset(Dataset):
    def __init__(self, image_dir, transform=None, normImage=False):
        self.image_dir = image_dir
        self.image_names = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.nii.gz')]
        self.transform = transform
        self.images = load_data_2D(self.image_names, normImage=normImage)  # Load all images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image

def save_sample_images(dataset, num_images=5, save_dir='saved_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    
    images = next(iter(dataloader))
    
    # Loop over images and save each one
    for i in range(num_images):
        image = images[i].numpy()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        save_path = os.path.join(save_dir, f'sample_image_{i}.png')
        plt.savefig(save_path)
        plt.close()  # Close figure after saving to avoid overwriting

dataset = HipMRIDataset(train_dir, normImage=True)

save_sample_images(dataset, num_images=5)