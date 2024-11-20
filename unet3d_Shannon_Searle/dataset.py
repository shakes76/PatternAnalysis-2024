# dataset.py
import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import rotate
from tqdm import tqdm
from scipy.ndimage import zoom
import random 

# Function to convert labels to one-hot encoded channels
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    # Get unique values (assuming categorical data)
    channels = np.unique(arr)
    # Create a result array with channels as the first dimension
    res = np.zeros((len(channels),) + arr.shape, dtype=dtype)
    
    # Loop over each unique category (channel)
    for c in channels:
        c = int(c)
        # Set the corresponding channel to 1 where the value matches the category
        res[c] = (arr == c).astype(dtype)
    
    return res

# Function to load 3D data
def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, early_stop=False): 
    num = len(image_names)
    first_case = nib.load(image_names[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # Remove extra dim
    
    if categorical:
        # Convert to categorical (one-hot) encoding with channels as the first dimension
        first_case = to_channels(first_case, dtype=dtype)
        channels, depth, height, width = first_case.shape
        images = np.zeros((num, channels, depth, height, width), dtype=dtype)  # [batch_size, channels, depth, height, width]
    else:
        depth, height, width = first_case.shape
        images = np.zeros((num, 4, depth, height, width), dtype=dtype)  # Non-categorical, assuming 4 channels

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
                # Store in correct order: [batch_size, channels, depth, height, width]
                images[i, :, :in_image.shape[1], :in_image.shape[2], :in_image.shape[3]] = in_image
            else:
                # For non-categorical, format is now [batch_size, 4, depth, height, width]
                images[i, :, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2]] = in_image

            if early_stop and i > 20:
                break
        except FileNotFoundError as e:
            print(f"Error loading image: {image_name}. {e}")

    return images


# Transformation to resize 3D volumes (depth, height, width)
class Resize3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Calculate zoom factors
        zoom_factors = [s / o for s, o in zip(self.size, image.shape[1:])]
        image = zoom(image, (1, *zoom_factors), order=1)  
        label = zoom(label, (1, *zoom_factors), order=0)  
        return {'image': image, 'label': label}

# Normalize the image (assuming image is already resized)
class Normalize3D: 
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Normalize the image
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std

        return {'image': image, 'label': label}

class RandomFlip3D:
    def __init__(self, axes = [0,1,2]):
        self.axes = axes  # Axes to flip (e.g., [0] for depth, [1] for height, [2] for width)

    def __call__(self, sample):
        images, labels = sample['image'], sample['label']
        
        # Randomly flip images and labels along specified axes
        if random.random() > 0.5:
            images = np.flip(images, axis=self.axes)
            labels = np.flip(labels, axis=self.axes)

        # Ensure the images and labels are contiguous
        images = np.ascontiguousarray(images)
        labels = np.ascontiguousarray(labels)

        return {'image': images, 'label': labels}
class RandomRotate3D:
    def __init__(self, angle_range):
        self.angle_range = angle_range

    def __call__(self, sample):
        images, labels = sample['image'], sample['label']
        
        # Rotate images and labels randomly
        angle_d = random.uniform(-self.angle_range, self.angle_range)  # Rotation angle for depth axis
        angle_h = random.uniform(-self.angle_range, self.angle_range)  # Rotation angle for height axis
        angle_w = random.uniform(-self.angle_range, self.angle_range)  # Rotation angle for width axis

        images = self.rotate_3d(images, (angle_d, angle_h, angle_w))
        labels = self.rotate_3d(labels, (angle_d, angle_h, angle_w))

        return {'image': images, 'label': labels}

    def rotate_3d(self, volume, angles):
        """
        Rotate the 3D volume by specified angles around each axis.

        Parameters:
        volume (ndarray): The 3D volume to rotate (shape: [C, D, H, W])
        angles (tuple): Angles for rotation around (depth, height, width) axes.

        Returns:
        ndarray: Rotated volume.
        """
        # Unpack angles
        angle_d, angle_h, angle_w = angles
        
        # Rotate the volume
        # The order of rotation can affect the final orientation. Adjust as needed.
        volume = rotate(volume, angle_d, axes=(2, 1), reshape=False)  # Rotate around depth (D)
        volume = rotate(volume, angle_h, axes=(1, 0), reshape=False)  # Rotate around height (H)
        volume = rotate(volume, angle_w, axes=(0, 2), reshape=False)  # Rotate around width (W)

        return volume
   
# Custom dataset class for PyTorch with transformations for 3D data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, img_dir, labels_dir, transform=None):
        self.image_filenames = image_filenames
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = load_data_3D([img_path])[0]  # Assuming load_data_3D returns a 3D image (depth, height, width)
        image = torch.tensor(image, dtype=torch.float32)

        label_filename = self.image_filenames[idx].replace("LFOV", "SEMANTIC_LFOV")  # replacement
        label_path = os.path.join(self.labels_dir, label_filename)
        label = load_data_3D([label_path], categorical=True)[0]
        label = torch.tensor(label, dtype=torch.float32)
        # Apply transformations if any
        if self.transform:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image = sample['image']
            label = sample['label']
        # Ensure the image and label are in the correct shape: (channels, depth, height, width)
        # The model expects input in [batch_size, channels, depth, height, width]
        # and labels in [batch_size, num_classes, depth, height, width]
        return image, label