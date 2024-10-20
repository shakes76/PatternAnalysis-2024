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


def process_slice(model, image_slice):
    # Convert the image slice to a PyTorch tensor
    image_tensor = torch.from_numpy(image_slice.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Make prediction for the slice
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()  # Shape: (H, W)

    return prediction


def main(image_path, mask_path, model_path):
    # Load the trained model
    model = UNet3D(1, 6)  # Initialize your model architecture
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Load the MRI image
    image_data = nib.load(image_path).get_fdata()
    num_slices = min(64, image_data.shape[2])  # Only process up to 64 slices to reduce memory usage

    # Normalize the image
    image_min, image_max = image_data.min(), image_data.max()
    image_data = (image_data - image_min) / (image_max - image_min) if image_max != image_min else image_data

    # Load the Mask image
    mask_data = nib.load(mask_path).get_fdata()[:, :, :num_slices]

    # Process and visualize slice by slice
    for slice_index in range(num_slices):
        print(f"Processing slice {slice_index + 1}/{num_slices}")
        image_slice = image_data[:, :, slice_index]

        # Make prediction for the current slice
        prediction_slice = process_slice(model, image_slice)

        # Visualise the current slice
        visualise_slice(image_data, mask_data, prediction_slice, slice_index)


if __name__ == "__main__":
    image_path = "/Users/shwaa/Downloads/HipMRI_study_complete_release_v1/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz"
    mask_path = "/Users/shwaa/Downloads/HipMRI_study_complete_release_v1/semantic_labels_anon/Case_004_Week0_SEMANTIC_LFOV.nii.gz"
    model_path = "/Users/shwaa/Desktop/model/new_model.pth"
    main(image_path, mask_path, model_path)
