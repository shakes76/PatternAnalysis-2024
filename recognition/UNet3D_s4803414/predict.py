import torch
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import MRIDataset
from modules import UNet3D


def visualise_slice(image, mask, prediction, slice_index):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image Slice")
    plt.imshow(image[:, :, slice_index], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Mask Slice")
    plt.imshow(mask[:, :, slice_index], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Prediction Slice")
    plt.imshow(prediction[:, :, slice_index], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(image_path, mask_path, model_path):
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

    # Make predictions
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.argmax(prediction, dim=1)  # Get predicted class
        prediction = prediction.squeeze().cpu().numpy()  # Shape: (256, 256, 128)

    # Load the Mask image
    mask_data = nib.load(mask_path).get_fdata()
    mask_data = image_data[:, :, :128]

    # Visualise slices
    slice_index = 64  # Choose a slice to visualize (middle slice)
    visualise_slice(image_data[0, :, :],
                    mask_data[0, :, :] if mask_data is not None else np.zeros_like(image_data[0, :, :]), prediction,
                    slice_index)

if __name__ == "__main__":
    image_path = "path_to_your_mri_image.nii.gz"
    mask_path = "path_to_your_mri_image.nii.gz"
    model_path = "model.pth"
    main(image_path, mask_path, model_path)
