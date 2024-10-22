import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load NIfTI file
nifti_file_path = 'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_train/seg_004_week_0_slice_0.nii.gz'
nifti_image = nib.load(nifti_file_path)
image_data = nifti_image.get_fdata()  # Get the image data as a NumPy array

# Check the shape of the loaded image
print(f'Image shape: {image_data.shape}')

# Display the image based on the shape
def display_image(image_data):
    if len(image_data.shape) == 2:  # If the image is 2D
        plt.imshow(image_data, cmap='gray')
        plt.title('Single 2D Slice')
        plt.axis('off')
        plt.show()
    elif len(image_data.shape) == 3:  # If the image is 3D
        num_images = image_data.shape[2]
        num_slices = min(num_images, 5)  # Ensure we don't exceed available slices
        slice_indices = np.linspace(0, num_images - 1, num_slices, dtype=int)

        plt.figure(figsize=(15, 5))
        for i, slice_idx in enumerate(slice_indices):
            plt.subplot(1, num_slices, i + 1)
            plt.imshow(image_data[:, :, slice_idx], cmap='gray')  # Display the axial slice
            plt.title(f'Slice {slice_idx}')
            plt.axis('off')  # Hide the axis

        plt.tight_layout()
        plt.show()
    else:
        print("Unsupported image dimensions")

# Call the function to display the image
display_image(image_data)
