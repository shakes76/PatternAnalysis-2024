import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scipy.ndimage

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

# Helper function to ensure all images are of the same size
def resize_image(inImage, target_shape):
    # Get the zoom factors for resizing
    zoom_factors = [
        target_shape[i] / inImage.shape[i] for i in range(len(target_shape))
    ]
    # Resize image using the zoom factors
    return scipy.ndimage.zoom(inImage, zoom_factors, order=1)  # Bilinear interpolation

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False, target_shape=(256, 128)):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    Parameters:
    - normImage: bool (normalize the image 0.0-1.0)
    - early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    - target_shape: Target shape to resize the images (rows, cols).
    '''
    affines = []
    
    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Sometimes extra dims, remove
    
    rows, cols = target_shape  # Use the target shape for consistency
    images = np.zeros((num, rows, cols), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        
        # Normalize image if required
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        # Resize image to the target shape
        inImage_resized = resize_image(inImage, target_shape)
        
        images[i, :, :] = inImage_resized
        
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

# Dataloader class for HipMRI data
class HipMRILoader:
    def __init__(self, train_dir, validate_dir, test_dir, batch_size=128, transform=None, num_workers=4):
        self.batch_size = batch_size
        self.transform = transform
        
        # Create datasets
        self.train_dataset = HipMRIDataset(train_dir, transform=self.transform, normImage=True)
        self.validate_dataset = HipMRIDataset(validate_dir, transform=self.transform, normImage=True)
        self.test_dataset = HipMRIDataset(test_dir, transform=None, normImage=True)  # No transforms for test data
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.validate_loader = DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def get_loaders(self):
        return self.train_loader, self.validate_loader, self.test_loader
    



# def save_sample_images(dataset, num_images=5, save_dir='saved_images'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    
#     images = next(iter(dataloader))
    
#     # Loop over images and save each one
#     for i in range(num_images):
#         image = images[i].numpy()
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
        
#         save_path = os.path.join(save_dir, f'sample_image_{i}.png')
#         plt.savefig(save_path)
#         plt.close()  # Close figure after saving to avoid overwriting

# dataset = HipMRIDataset(train_dir, normImage=True)

# save_sample_images(dataset, num_images=5)

# print("Start")
# # Specify the file path
# file_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train/case_012_week_6_slice_25.nii.gz'

# # Load the NIfTI file
# image_data = nib.load(file_path).get_fdata(caching='unchanged')

# # Check the shape of the image
# print("Shape of the image:", image_data.shape)

# # Load the single image using load_data_2D
# img = load_data_2D([file_path])  # Pass a list with the file path

# # Extract the image slice (assuming it's a single slice)
# image_slice = img[0]  # This gives you the 2D array of the first (and only) image

# # Save the image using matplotlib
# plt.imshow(image_slice, cmap='gray')  # Use 'gray' colormap for grayscale images
# plt.axis('off')  # Hide axis
# plt.savefig('case_012_week_6_slice_25.png', bbox_inches='tight', pad_inches=0)  # Save without padding
# plt.close()  # Close the plot


# print("End")
