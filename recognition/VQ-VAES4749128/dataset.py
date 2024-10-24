import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import os
import numpy as np

from PIL import Image

import os
import gzip
import shutil
import nibabel as nib



# class GrayscaleImageDataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_dir = image_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
#         # print(self.image_paths)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path) 
#         if self.transform:
#             image = self.transform(image)
#         return image

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        res[..., c] = (arr == c) * 1
    return res


class GrayscaleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.nii')]
        # print(self.image_paths)

    def __len__(self):
        return len(self.image_paths)
    
    def __read_nifti__(self, filepath):
        niftiImage = nib.load(filepath)
        # print(filepath)
        inImage = niftiImage.get_fdata(caching='unchanged')
        dtype=np.float32
        inImage = inImage.astype(dtype)
        inImage = 255.0 * (inImage - inImage.min()) / inImage.ptp()
        return inImage.astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # image = Image.open(img_path) # pillow image type
        image = self.__read_nifti__(img_path)

        if self.transform:
            image = self.transform(image)
        return image


def load_dataset(args):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Resize((256, 128))
    ])

    #load training set
    train_dataset = GrayscaleImageDataset(image_dir=args.train_dir, transform=transform)
    #load val set
    test_dataset = GrayscaleImageDataset(image_dir=args.test_dir, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')


