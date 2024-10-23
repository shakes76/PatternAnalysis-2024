from modules import UNet
from dataset import MedicalImageDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")


def dice_coefficient(y_true, y_pred, epsilon=1e-8):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    dice = (2. * intersection + epsilon) / (tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) + epsilon)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = 0
    d_loss = dice_loss(y_true, y_pred)
    total_loss = bce + d_loss
    return total_loss


model = UNet(input_dims=(256, 144, 1))

# Initialize learning rate scheduler
decay_steps = 1000
initial_learning_rate = 1e-4 # Previous: 0.01, 1e-5
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps)

# Adam Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, clipvalue=1.0, clipnorm=1.0)

model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient])

base_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/"
image_dir = os.path.join(base_dir, "keras_slices_train") 
mask_dir = os.path.join(base_dir, "keras_slices_seg_train")

train_dataset = MedicalImageDataset(image_dir=image_dir, mask_dir=mask_dir, normImage=True, batch_size=8, shuffle=True)
train_dataset.print_mask_categories()
dataset = train_dataset.get_dataset()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    dataset, 
    epochs=5, 
    steps_per_epoch=len(dataset),
    callbacks=[tensorboard_callback],
    verbose=1
)

model.summary()
model.save('unet_model', save_format='tf')


def plot_sample_images(dataset, model):
    for images, masks in dataset.take(1):  # Get a batch from the dataset
        print(f"Shape of images: {images.shape}")
        print(f"Shape of masks: {masks.shape}")   

        predictions = model.predict(images)

        image = images[0]  # Actual image
        mask = masks[0]    # Ground truth mask

        # Convert predicted mask to binary (threshold at 0.5)
        pred_mask = predictions[0] 
        pred_mask = (pred_mask > 0.5).astype(np.float32)  # Thresholding to convert to binary mask

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


plot_sample_images(dataset, model)
