import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
"""
Brief:
Segment the (downsampled) Prostate 3D data set (see Appendix for link) with the 3D Improved UNet3D [3] 
with all labels having a minimum Dice similarity coefficient of 0.7 on the test set. See also CAN3D [4] 
for more details and use the data augmentation library here for TF or use the appropriate transforms in PyTorch. 
You may begin with the original 3D UNet [5]. You will need to load Nifti file format and sample code is provided in Appendix B. 
[Normal Difficulty- 3D UNet] [Hard Difficulty- 3D Improved UNet]

"""
# Function to convert labels to one-hot encoded channels
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load 3D Nifti images
def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, early_stop=False):
    num = len(image_names)
    first_case = nib.load(image_names[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # Remove extra dim
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, image_name in enumerate(tqdm(image_names)):
        nifti_image = nib.load(image_name)
        in_image = nifti_image.get_fdata(caching='unchanged')
        
        if len(in_image.shape) == 4:
            in_image = in_image[:, :, :, 0]  # Remove extra dim
        in_image = in_image.astype(dtype)
        
        if norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()  # Normalization
        
        if categorical:
            in_image = to_channels(in_image, dtype=dtype)
            images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2], :] = in_image  # With padding if necessary
        else:
            images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2]] = in_image  # With padding if necessary
        
        if early_stop and i > 20:
            break

    return images

# Custom dataset class for PyTorch with transformations for 3D data
class CustomDataset(Dataset):
    def __init__(self, image_filenames, labels, img_dir, labels_dir, transform=None):
        self.image_filenames = image_filenames
        self.labels = labels
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = load_data_3D([img_path])[0]
        
        label_filename = self.image_filenames[idx].replace("MR", "SEMANTIC_LFOV")
        label_path = os.path.join(self.labels_dir, label_filename)
        label = nib.load(label_path).get_fdata()

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Convert to tensor
        label = torch.tensor(label, dtype=torch.float32)  # Ensure label is tensor
        return torch.tensor(image, dtype=torch.float32), label

# Define custom transformation for 3D data
class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

class Resize3D:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, volume):
        # Resize volume to the new shape
        return np.resize(volume, self.new_shape)

class Normalize3D:
    def __call__(self, volume):
        # Normalize the volume
        return (volume - np.mean(volume)) / np.std(volume)

# Example usage of the dataset with transformations
def main():
    img_dir = "Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
    labels_dir = "Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"
    
    image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]
    
    # Define transformations
    transform = Compose3D([
        Resize3D((128, 128, 64)),  # Resize to (depth, height, width)
        Normalize3D()  # Normalize
    ])

    # Create dataset
    dataset = CustomDataset(image_filenames, labels=None, img_dir=img_dir, labels_dir=labels_dir, transform=transform)

    # DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


if __name__ == "__main__":
    main()