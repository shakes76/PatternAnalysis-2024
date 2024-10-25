import os
import torch
import nibabel as nib
import numpy as np
from torchvision.transforms.functional import resize

def preprocess_and_save(input_dir, output_dir, target_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            print(f"Processing {filename}...")
            # Load the .nii.gz file
            image = nib.load(os.path.join(input_dir, filename)).get_fdata()
            image = np.expand_dims(image, axis=0)  # Add channel dimension

            # Convert to a PyTorch tensor and resize
            tensor_image = torch.tensor(image, dtype=torch.float32)
            tensor_image = resize(tensor_image, target_size)  # Enforce consistent size

            # Save as .pt file
            output_path = os.path.join(output_dir, filename.replace('.nii.gz', '.pt'))
            torch.save(tensor_image, output_path)

    print("Preprocessing complete.")

# Example usage
input_dir = r"drive/MyDrive/keras_slices_train"
output_dir = r"1HipMRI_study_keras_slices_data/keras_slices_train"

preprocess_and_save(input_dir, output_dir)
