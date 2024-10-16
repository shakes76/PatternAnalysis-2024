import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm
import os
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
torch.autograd.set_detect_anomaly(True)

# Custom dataset class (similar to the one discussed previously)
class NiftiDataset(Dataset):
    def __init__(self, image_paths, transform=None, normImage=False, categorical=False, dtype=np.float32, early_stop=False):
        self.image_paths = image_paths
        self.transform = transform
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.early_stop = early_stop
        self.images = self.load_data_2D(image_paths, normImage=self.normImage, categorical=self.categorical, dtype=self.dtype, early_stop=self.early_stop)

    def load_data_2D(self, image_names, normImage=False, categorical=False, dtype=np.float32, early_stop=False):
        num = len(image_names)
        first_case = nib.load(image_names[0]).get_fdata(caching='unchanged')
        if len(first_case.shape) == 3:
            first_case = first_case[:, :, 0]  # Remove extra dimension
        rows, cols = first_case.shape[:2]
        if categorical:
            channels = first_case.shape[2]
            images = np.zeros((num, rows, cols, channels), dtype=dtype)
        else:
            images = np.zeros((num, rows, cols), dtype=dtype)

        for i, image_name in enumerate(tqdm(image_names)):

            nifti_image = nib.load(image_name)
            image_data = nifti_image.get_fdata(caching='unchanged')
            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]
            image_data = image_data.astype(dtype)
            
            if image_data.shape[1] != 256 or image_data.shape[2] != 128:
                image_data = self.resize_image(image_data)

            if normImage:
                image_data = (image_data - image_data.mean()) / image_data.std()

            if categorical:
                image_data = self.to_channels(image_data, dtype=dtype)
                images[i, :, :, :] = image_data
            else:
                images[i, :, :] = image_data

            if i > 20 and early_stop:
                break

        return images
    def resize_image(self, image_data):
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, ...]
        if image_data.ndim == 3:
            image_data = F.resize(torch.tensor(image_data), (256, 128))
            return image_data.numpy()
        else:
            raise ValueError(f"Unsupported image data shape: {image_data.shape}")

    def to_channels(self, arr, dtype=np.uint8):
        channels = np.unique(arr)
        res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
        for c in channels:
            c = int(c)
            res[..., c:c + 1][arr == c] = 1
        return res

    def __len__(self):
        return len(self.image_paths)

    def load_nifti(self, img_paths):
        img = nib.load(img_paths)
        img_data = img.get_fdata()
        return img_data

    def __getitem__(self, idx):
        image = self.load_nifti(self.image_paths[idx])
        image = self.images[idx]
        if self.transform:
            image = torch.from_numpy(image).unsqueeze(0)
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32)


    # Function to return 3 dataloaders: train, validation, and test
    @staticmethod
    def get_dataloaders(train_dir, val_dir, test_dir, batch_size=8, num_workers=4, transform=None, normImage=True):
        # List of paths for each set
        train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.nii.gz')]
        val_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.nii.gz')]
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.nii.gz')]

        # Datasets
        train_dataset = NiftiDataset(image_paths=train_images, transform=transform, normImage=normImage)
        val_dataset = NiftiDataset(image_paths=val_images, transform=transform, normImage=normImage)
        test_dataset = NiftiDataset(image_paths=test_images, transform=transform, normImage=normImage)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader

# Example usage:
#train_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_train"
#val_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_validate"
#test_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_test"

#base_dir = os.getcwd()
#train_dir = os.path.join(base_dir, train_dir)
#val_dir = os.path.join(base_dir, val_dir)
#test_dir = os.path.join(base_dir, test_dir)

# Transforms (optional)
data_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

# Get the DataLoaders
def get_dataloaders(train_dir, val_dir, test_dir, batch_size=8, num_workers=4, transform=None, normImage=True):
    # List of paths for each set
    train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.nii.gz')]
    val_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.nii.gz')]
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.nii.gz')]

    # Datasets
    train_dataset = NiftiDataset(image_paths=train_images, transform=transform, normImage=normImage)
    val_dataset = NiftiDataset(image_paths=val_images, transform=transform, normImage=normImage)
    test_dataset = NiftiDataset(image_paths=test_images, transform=transform, normImage=normImage)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# train_loader, val_loader, test_loader = get_dataloaders(train_dir, val_dir, test_dir, batch_size=16, num_workers=4, transform=data_transforms)
#print("Success! - Made all the loaders from the directories")

#TODO: Number of workers may need to decrease to 1 to make it faster



