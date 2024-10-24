from modules import UNet
from dataset import MedicalImageDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
import argparse

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")


def dice_coefficient(y_true, y_pred, epsilon=1e-8):
    """
    Calculate the Dice coefficient between the true and predicted masks.

    Args:
        y_true (tf.Tensor): Ground truth mask.
        y_pred (tf.Tensor): Predicted mask.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        tf.Tensor: Mean Dice coefficient over the batch.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    dice = (2. * intersection + epsilon) / (tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) + epsilon)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss, which is 1 minus the Dice coefficient.

    Args:
        y_true (tf.Tensor): Ground truth mask.
        y_pred (tf.Tensor): Predicted mask.

    Returns:
        tf.Tensor: Dice loss value.
    """
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    Returns the loss as the Dice loss.

    Args:
        y_true (tf.Tensor): Ground truth mask.
        y_pred (tf.Tensor): Predicted mask.

    Returns:
        tf.Tensor: Combined loss value.
    """
    bce = 0  # Binary Cross Entropy placeholder, set to 0 for simplicity
    d_loss = dice_loss(y_true, y_pred)
    total_loss = bce + d_loss
    return total_loss


def plot_sample_images(dataset, model):
    """
    Plot sample images along with their ground truth and predicted masks.

    Args:
        dataset: TensorFlow dataset containing images and masks.
        model: Trained UNet model for prediction.
    """
    for images, masks in dataset.take(1):  # Get a batch from the dataset
        predictions = model.predict(images)

        image = images[0]  # Actual image
        mask = masks[0]    # Ground truth mask

        # Convert predicted mask to binary (threshold at 0.5)
        pred_mask = predictions[0] 
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        # Plot the actual image, ground truth, and predicted mask
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
        plt.show()


def main(base_dir, image_dir, mask_dir):
    """
    Main function to set up and train the UNet model for medical image segmentation.

    Args:
        base_dir (str): Base directory containing the image and mask directories.
        image_dir (str): Directory containing the input images.
        mask_dir (str): Directory containing the corresponding masks.
    """
    model = UNet(input_dims=(256, 144, 1))

    # Initialize learning rate scheduler
    decay_steps = 1000
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)

    # Initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, clipvalue=1.0, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient])

    # Set up directories for images and masks
    image_dir = os.path.join(base_dir, image_dir)
    mask_dir = os.path.join(base_dir, mask_dir)

    # Create training dataset
    train_dataset = MedicalImageDataset(image_dir=image_dir, mask_dir=mask_dir, normImage=True, batch_size=8, shuffle=True)
    dataset = train_dataset.get_dataset()

    # Set directory to store TensorBoard logs
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train model
    model.fit(
        dataset, 
        epochs=25, 
        steps_per_epoch=len(dataset),
        callbacks=[tensorboard_callback],
        verbose=1
    )

    # Give basic info on model features
    model.summary()
    # Save model
    model.save('unet_model', save_format='tf')
    # Plot model output
    plot_sample_images(dataset, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model for medical image segmentation.")
    
    # Default paths
    default_base_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/"
    default_image_dir = "keras_slices_train"
    default_mask_dir = "keras_slices_seg_train"

    # Adding arguments with default values
    parser.add_argument('--base_dir', type=str, default=default_base_dir, help='Base directory for image and mask data.')
    parser.add_argument('--image_dir', type=str, default=default_image_dir, help='Directory containing images.')
    parser.add_argument('--mask_dir', type=str, default=default_mask_dir, help='Directory containing masks.')

    args = parser.parse_args()
    main(args.base_dir, args.image_dir, args.mask_dir)
