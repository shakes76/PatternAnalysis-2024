import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
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

# Load saved UNet model and desired image/mask folder
model = tf.keras.models.load_model('unet_model', custom_objects={'dice_coefficient': dice_coefficient})
base_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/"
test_image_dir = os.path.join(base_dir, "keras_slices_test") 
test_mask_dir = os.path.join(base_dir, "keras_slices_seg_test")
test_dataset = MedicalImageDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, normImage=True, batch_size=8, shuffle=False)
test_loader = test_dataset.get_dataset()

# Function to save sample images
def plot_sample_images(dataset, model):
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
            # Save the plot as a PNG file at current dir
            plt.savefig(f"sample_image_{i}_{idx}.png", format='png')


plot_sample_images(test_loader, model)
