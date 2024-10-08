import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

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
        try:
            nifti_image = nib.load(image_name)
            in_image = nifti_image.get_fdata(caching='unchanged')

            if len(in_image.shape) == 4:
                in_image = in_image[:, :, :, 0]  # Remove extra dim
            in_image = in_image.astype(dtype)

            if norm_image:
                in_image = (in_image - in_image.mean()) / in_image.std()  # Normalization
            
            if categorical:
                in_image = to_channels(in_image, dtype=dtype)
                images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2], :] = in_image
            else:
                images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2]] = in_image

            if early_stop and i > 20:
                break
                
        except FileNotFoundError as e:
            print(f"Error loading image: {image_name}. {e}")

    return images

# Custom dataset class for PyTorch with transformations for 3D data
class CustomDataset(Dataset):
    def __init__(self, image_filenames, img_dir, labels_dir, transform=None):
        self.image_filenames = image_filenames
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

        try:
            label = nib.load(label_path).get_fdata()
        except FileNotFoundError as e:
            print(f"Error loading label: {label_path}. {e}")
            label = np.zeros(image.shape[-3:])  # Create a blank label if not found

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define custom transformations for 3D data
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
        return np.resize(volume, self.new_shape)

class Normalize3D:
    def __call__(self, volume):
        return (volume - np.mean(volume)) / np.std(volume)

# Main function
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
    dataset = CustomDataset(image_filenames, img_dir, labels_dir, transform=transform)

    # DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in data_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()
