import zipfile
import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "HipMRI_study_keras_slices_data"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Unzip file
    with zipfile.ZipFile("/content/HipMRI_study_keras_slices_data.zip", "r") as zip_ref:
        print("Unzipping keras_png_slices_data...")
        zip_ref.extractall(image_path)

# Function to convert array to one-hot encoded channels (useful for segmentation tasks)
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res

# Load medical image data and prepare it for deep learning models
class HipMRIDataset(Dataset):
    def __init__(self, root_dir, subset='train', normImage=False, categorical=False, dtype=np.float32, target_size=(256, 256)):
        '''
        Load medical image data from the given directory.

        Args:
        - root_dir (str): Root directory containing subdirectories like keras_slices_train, keras_slices_validate, etc.
        - subset (str): Which subset to use ('train', 'validate', 'test').
        - normImage (bool): Whether to normalize images (mean=0, std=1).
        - categorical (bool): Whether to convert images to categorical (one-hot).
        - dtype (type): Data type for the output images.
        - target_size (tuple): The size to which all images will be resized (height, width).
        '''
        self.root_dir = root_dir
        self.subset = subset
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.target_size = target_size
        self.resize = Resize(target_size)

        # Construct the full path based on the subset
        self.data_dir = os.path.join(root_dir, f'keras_slices_{subset}')
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Path '{self.data_dir}' does not exist.")

        # Collect all files from the directory
        self.image_files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.nii') or file.endswith('.nii.gz')]
        if len(self.image_files) == 0:
            raise ValueError(f"No suitable image files found in directory '{self.data_dir}'.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files):
            raise IndexError("Index out of bounds")

        # Load the NIfTI image
        image_path = self.image_files[idx]
        nifti_image = nib.load(image_path)
        inImage = nifti_image.get_fdata(caching='unchanged')

        # Handle 3D images: use the first slice if it's a 3D volume
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        # Convert to specified data type
        inImage = inImage.astype(self.dtype)

        # Normalize the image if needed (mean=0, std=1)
        if self.normImage:
            inImage = (inImage - np.mean(inImage)) / np.std(inImage)

        # Convert to categorical format if needed (e.g., for segmentation tasks)
        if self.categorical:
            inImage = to_channels(inImage, dtype=self.dtype)

        # Convert image to torch tensor
        inImage = torch.tensor(inImage, dtype=torch.float32)

        # Add channel dimension if it's missing
        if inImage.dim() == 2:
            inImage = inImage.unsqueeze(0)

        # Resize the image to the target size
        inImage = self.resize(inImage)

        return inImage

# DataLoader function to handle batching
def get_data_loader(root_dir, subset='train', batch_size=16, shuffle=True, normImage=False, categorical=False, dtype=np.float32, target_size=(256, 256)):
    dataset = HipMRIDataset(root_dir, subset=subset, normImage=normImage, categorical=categorical, dtype=dtype, target_size=target_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

# Make sure data_loader work properly
""" import torch
import matplotlib.pyplot as plt

# set path and bacth size
data_dir = image_path
batch_size = 4

data_loader = get_data_loader(data_dir, batch_size=batch_size, shuffle=True, normImage=True)

def visualize_batch(data_loader):
    for batch in data_loader:
        fig, axs = plt.subplots(1, batch_size, figsize=(15, 5))
        for i in range(batch_size):
            if batch.ndim == 4:  # batch shape is (batch_size, channels, height, width)
                image = batch[i, 0, :, :]
            elif batch.ndim == 3:  # if (batch_size, height, width)
                image = batch[i, :, :]
            else:
                raise ValueError("Unexpected data shape")

            axs[i].imshow(image.numpy(), cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Image {i + 1}")

        plt.show()
        break

visualize_batch(data_loader) """

