import torch
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import MRIDataset
from modules import UNet3D

def main(image_path, model_path):
    # Load the trained model
    model = UNet3D(1, 6)  # Initialize your model architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load the MRI image
    image_data = nib.load(image_path).get_fdata()
    image_data = image_data[:, :, :128]  # Crop to the first 128 slices

    # Normalize the image
    image_min, image_max = image_data.min(), image_data.max()
    image_data = (image_data - image_min) / (image_max - image_min) if image_max != image_min else image_data
    image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(image_data.astype(np.float32)).unsqueeze(0)  # Shape: (1, 1, 256, 256, 128)
