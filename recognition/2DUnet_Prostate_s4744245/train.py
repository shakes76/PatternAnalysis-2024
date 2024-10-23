import numpy as np
import random


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K





# plot the training/validation curves from the history
import matplotlib.pyplot as plt

#define training variables
BATCH_SIZE = 1
EPOCHS = 1
n_classes = 5

# Define the Dice similarity coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten ground truth tensor
    y_pred_f = K.flatten(y_pred)  # Flatten predicted tensor
    intersection = K.sum(y_true_f * y_pred_f)  # Intersection between true and predicted
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_dice_loss(class_weights):
    def loss(y_true, y_pred):
        dice_scores = []
        for i in range(n_classes):
            dice_scores.append(dice_coefficient(y_true[..., i], y_pred[..., i]))
        dice_scores = K.stack(dice_scores)  # Shape: (n_classes,)

        # Ensure class_weights is a tensor of shape (n_classes, 1) for correct broadcasting
        class_weights_tensor = K.constant(np.array(class_weights).reshape(-1, 1))  # Shape: (n_classes, 1)

        # Calculate the weighted Dice score
        weighted_dice = K.dot(class_weights_tensor, K.reshape(dice_scores, (-1, 1)))  # Shape: (1, 1)

        return 1 - K.flatten(weighted_dice)  # Return as a scalar
    return loss

# Define the Dice loss 
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def train_unet_model(model, train_images, train_labels, 
                     val_images, val_labels, 
                     batch_size=BATCH_SIZE, epochs=EPOCHS, model_save_path="unet_model.h5", class_weights=[1, 1, 1, 1, 1]):
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
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[dice_coefficient])

    # Print the model summary to verify the architecture
    model.summary()

    # Train the model
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint, early_stopping])
    
    #plot_training(history)

    return history



def plot_training(history):
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Model Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
        # Save the figure
    plt.savefig('training_curve.png')
    plt.close()

