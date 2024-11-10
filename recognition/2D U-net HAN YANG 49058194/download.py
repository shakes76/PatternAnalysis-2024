"""
download.py
-----------
This script downloads and preprocesses MRI data for training a segmentation model.

Usage:
    Run this script to download and prepare the dataset for model training.

Author: Han Yang
Date: 24/09/2024
"""
import os
import requests
import zipfile
import io
import nibabel as nib
from skimage.transform import resize
import numpy as np

# Download and extract data
def download_and_extract(url, extract_to='HipMRI_project_data'):
    """ 
    Downloads the dataset from the specified URL and extracts it to the given directory. 
    Args: 
        url (str): The URL to download the dataset from. 
        save_dir (str): Directory to save the downloaded files. 
    """ 
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall(extract_to)

# Process nii files in npy format
def load_and_process_nii_files(root_dir, save_dir, target_size=(128, 128)):
    """ 
    Loads NIfTI files, processes them, and saves as .npy files. 
    Args:
        root_dir (str): Directory containing NIfTI files. 
        save_dir (str): Directory to save processed .npy files. 
        target_size (tuple): Target size for resizing images. 
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Traverse each folder to find. nii files
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith('.nii') or file_name.lower().endswith('.nii.gz'):
                file_path = os.path.join(root, file_name)
                print(f"Loading {file_path}")

                # Using nibabel to load nii files
                nii_img = nib.load(file_path)
                img_data = nii_img.get_fdata()
                print(f"Image shape: {img_data.shape}")  # Print the shape of the image to check its dimensions
                
                if len(img_data.shape) == 3:  # 3D images
                    for i in range(img_data.shape[2]):  # For each slice
                        slice_2d = img_data[:, :, i]
                        resized_slice = resize(slice_2d, target_size, preserve_range=True)
                        save_path = os.path.join(save_dir, f"{file_name}_slice_{i}.npy")
                        np.save(save_path, resized_slice)
                        print(f"Saved {save_path}")
                elif len(img_data.shape) == 2:  # 2D images
                    resized_slice = resize(img_data, target_size, preserve_range=True)
                    save_path = os.path.join(save_dir, f"{file_name}_slice.npy")
                    np.save(save_path, resized_slice)
                    print(f"Saved {save_path}")
                else:
                    print(f"Unexpected shape for file {file_name}: {img_data.shape}")


if __name__ == "__main__":
    # download url
    url = "https://filesender.aarnet.edu.au/download.php?token=76f406fd-f55d-497a-a2ae-48767c8acea2&files_ids=23102543"

    # Extract and processing
    root_dir = 'HipMRI_study_keras_slices_data'  # Final path to save the. nii file
    processed_dir = os.path.join(root_dir, 'processed_nii_files')  # The path to save the. npy file

    # Download and extract files
    download_and_extract(url, root_dir)

    # Load and process nii files
    load_and_process_nii_files(root_dir, processed_dir)