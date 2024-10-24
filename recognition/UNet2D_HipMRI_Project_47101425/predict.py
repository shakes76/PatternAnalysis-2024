import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from modules import UNet
from dataset import MedicalImageDataset
from train import dice_coefficient

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")


def plot_sample_images(dataset, model):
    """
    Plot and save sample images along with their ground truth and predicted masks.

    Args:
        dataset: TensorFlow dataset containing test images and masks.
        model: Loaded UNet model for making predictions.
    """
    for i, (images, masks) in enumerate(dataset.take(1)): 
        print(f"Shape of images: {images.shape}")
        print(f"Shape of masks: {masks.shape}")

        predictions = model.predict(images)

        for idx in range(images.shape[0]):
            image = images[idx]  
            mask = masks[idx]    
            pred_mask = predictions[idx] 

            pred_mask = (pred_mask > 0.5).astype(np.float32)

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            # Plot actual image
            ax[0].imshow(image, cmap='gray')
            ax[0].set_title("Actual Image")
            ax[0].axis('off')

            # Plot ground truth mask
            ax[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis('off')

            # Plot predicted mask
            ax[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            ax[2].set_title("Predicted Mask")
            ax[2].axis('off')

            plt.tight_layout()
            # Save the plot as a PNG file in the current directory
            plt.savefig(f"sample_image_{i}_{idx}.png", format='png')
            plt.close()  # Close the figure to free up memory


def main(model_path, base_dir, image_dir, mask_dir):
    """
    Load a saved UNet model and predict masks for test images.

    Args:
        model_path (str): Path to the saved UNet model.
        base_dir (str): Base directory containing the image and mask directories.
        image_dir (str): Directory containing the test images.
        mask_dir (str): Directory containing the corresponding test masks.
    """
    # Load saved UNet model
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})

    # Set up directories for test images and masks
    test_image_dir = os.path.join(base_dir, image_dir)
    test_mask_dir = os.path.join(base_dir, mask_dir)

    test_dataset = MedicalImageDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, normImage=True, batch_size=8, shuffle=False)
    test_loader = test_dataset.get_dataset()

    # Function to save sample images
    plot_sample_images(test_loader, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load UNet model and predict masks for medical images.")
    
    # Default paths
    default_model_path = 'unet_model'
    default_base_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/"
    default_image_dir = "keras_slices_validate"
    default_mask_dir = "keras_slices_seg_validate"

    # Adding arguments with default values
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to the saved UNet model.')
    parser.add_argument('--base_dir', type=str, default=default_base_dir, help='Base directory for image and mask data.')
    parser.add_argument('--image_dir', type=str, default=default_image_dir, help='Directory containing test images.')
    parser.add_argument('--mask_dir', type=str, default=default_mask_dir, help='Directory containing test masks.')

    args = parser.parse_args()
    main(args.model_path, args.base_dir, args.image_dir, args.mask_dir)
