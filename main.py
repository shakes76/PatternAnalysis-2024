import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the paths to your image and label directories
image_dir = r'D:\final\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon'
label_dir = r'D:\final\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon'


def unet_3d(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Encoder path
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)

    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)

    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c4)

    # Decoder path
    u5 = layers.UpSampling3D((2, 2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling3D((2, 2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling3D((2, 2, 2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c7)

    # Output layer
    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(c7)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Get lists of image and label files
image_files = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
label_files = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))

# Ensure correspondence
assert len(image_files) == len(label_files), "Mismatch in number of images and labels."

# Select the first image and label
image_file = image_files[0]
label_file = label_files[0]

print(f"Selected Image File: {os.path.basename(image_file)}")
print(f"Selected Label File: {os.path.basename(label_file)}")

# Load the image
image_nib = nib.load(image_file)
image_data = image_nib.get_fdata()

# Load the label
label_nib = nib.load(label_file)
label_data = label_nib.get_fdata()

# Check shapes
print(f"Image shape: {image_data.shape}")
print(f"Label shape: {label_data.shape}")


# Choose a slice index (e.g., middle slice)
slice_index = image_data.shape[2] // 2

# Extract the slice
image_slice = image_data[:, :, slice_index]
label_slice = label_data[:, :, slice_index]

# Plot the image and label
plt.figure(figsize=(12, 6))

# Image slice
plt.subplot(1, 2, 1)
plt.imshow(image_slice.T, cmap='gray', origin='lower')
plt.title('MRI Image Slice')
plt.axis('off')

# Image with label overlay
plt.subplot(1, 2, 2)
plt.imshow(image_slice.T, cmap='gray', origin='lower')
plt.imshow(label_slice.T, cmap='jet', alpha=0.5, origin='lower')
plt.title('Image with Label Overlay')
plt.axis('off')

plt.show()
