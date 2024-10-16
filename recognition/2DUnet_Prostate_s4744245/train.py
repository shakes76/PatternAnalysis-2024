import numpy as np
from matplotlib import pyplot as plt
import glob

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

import os


from dataset import load_data_2D
from modules import unet_model


# Get all file paths for train, test, and validate sets
#train_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_train/*.nii.gz')
#test_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_test/*.nii.gz')
#validate_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_validate/*.nii.gz')

#seg_train_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_train/*.nii.gz')
#seg_test_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_test/*.nii.gz')
#seg_validate_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_validate/*.nii.gz')

train_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train/*.nii.gz')
test_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test/*.nii.gz')
validate_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate/*.nii.gz')

seg_train_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/*.nii.gz')
seg_test_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test/*.nii.gz')
seg_validate_files = glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/*.nii.gz')

# Load the images using the load_data_2D function
images_train = load_data_2D(train_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_test = load_data_2D(test_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_validate = load_data_2D(validate_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))

images_seg_train = load_data_2D(seg_train_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_seg_test = load_data_2D(seg_test_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))
images_seg_validate = load_data_2D(seg_validate_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128))

# print the shapes of the loaded datasets
print(f"Training data shape: {images_train.shape}")
print(f"Test data shape: {images_test.shape}")
print(f"Validation data shape: {images_validate.shape}")
print(f"Segement Training data shape: {images_seg_train.shape}")
print(f"Segement Test data shape: {images_seg_test.shape}")
print(f"Segement Validation data shape: {images_seg_validate.shape}")

#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Initialize the U-Net model with the adjusted input size
model = unet_model(input_size=(256, 128, 1))

# Define the Dice similarity coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten ground truth tensor
    y_pred_f = K.flatten(y_pred)  # Flatten predicted tensor
    intersection = K.sum(y_true_f * y_pred_f)  # Intersection between true and predicted
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define the Dice loss (if you want to use it as a loss function as well)
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


#temp compile to check model summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coefficient])

# Print the model summary to verify the architecture
#model.summary()


def train_unet_model(model, 
                     train_images, train_labels, 
                     val_images, val_labels, 
                     batch_size=8, epochs=50, model_save_path="unet_model.h5"):
    """
    Trains the U-Net model with the given training and validation data.

    Args:
        model: U-Net model
        train_images: Numpy array of training images
        train_labels: Numpy array of training labels
        val_images: Numpy array of validation images
        val_labels: Numpy array of validation labels
        batch_size: Size of the batches of data (default is 8)
        epochs: Number of epochs to train the model (default is 50)
        model_save_path: File path to save the trained model (default is 'unet_model.h5')

    Returns:
        history: Training history, useful for plotting and analyzing the performance
    """
    
    # Callbacks
    # ModelCheckpoint: Save the best model based on validation Dice coefficient
    checkpoint = ModelCheckpoint(model_save_path, 
                                 monitor='val_dice_coefficient', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')

    # EarlyStopping: Stop training if validation Dice coefficient does not improve
    early_stopping = EarlyStopping(monitor='val_dice_coefficient', 
                                   patience=10, 
                                   mode='max', 
                                   verbose=1)
    
    # Compile the model (ensure it's compiled with the desired loss and metric)
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient])
    
    # Train the model
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint, early_stopping])
    
    return history


# Create the U-Net model
model = unet_model(input_size=(256, 128, 1))

# Train the U-Net model
history = train_unet_model(model, 
                           images_train, images_seg_train, 
                           images_validate, images_seg_validate, 
                           batch_size=8, 
                           epochs=5, 
                           model_save_path="best_unet_model.h5")

# Optionally, you can plot the training/validation curves from the history
import matplotlib.pyplot as plt

plt.plot(history.history['dice_coefficient'])
plt.plot(history.history['val_dice_coefficient'])
plt.title('Model Dice Coefficient')
plt.ylabel('Dice Coefficient')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Function to save a validation image
def save_validation_image(image, mask, prediction, index):
    """Saves the original image, mask, and prediction."""
    
    # Create a figure
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')  # Use 'gray' for single-channel images
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')  # Use 'gray' for single-channel images
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(prediction.squeeze(), cmap='gray')  # Use 'gray' for single-channel images
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # Save the figure
    plt.savefig(f'validation_image_{index}.png')
    plt.close()

index = 0
image = images_validate[index]  # Shape (256, 128, 1)
mask = images_seg_validate[index]   # Shape (256, 128, 1)

# Get prediction from the model
prediction = model.predict(image[np.newaxis, ...])  # Add batch dimension, shape becomes (1, 256, 128, 1)

# Save the images
save_validation_image(image, mask, prediction, index)