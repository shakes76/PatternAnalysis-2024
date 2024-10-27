import numpy as np
import random
#import wandb

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


# plot the training/validation curves from the history
import matplotlib.pyplot as plt

#define training variables
BATCH_SIZE = 16
EPOCHS = 50
n_classes = 6
learning_rate = 0.0001

# Define the Dice similarity coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten ground truth tensor
    y_pred_f = K.flatten(y_pred)  # Flatten predicted tensor
    intersection = K.sum(y_true_f * y_pred_f)  # Intersection between true and predicted
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def weighted_categorical_crossentropy(class_weights):
    def wcce(y_true, y_pred):
        Kweights = K.constant(class_weights)

        if not tf.is_tensor(y_pred): 
            y_pred = K.constant(y_pred)
            
        y_true = K.cast(y_true, y_pred.dtype)
            
        # Calculate categorical cross-entropy and apply the class weights
        loss = K.categorical_crossentropy(y_true, y_pred)
        return loss * K.sum(y_true * Kweights, axis=-1)
        
    return wcce
    
def weighted_dice_loss(class_weights):
    def wdl(y_true, y_pred):
        # Calculate the Dice coefficient for each class
        dice_scores = []
        for i in range(n_classes):  # Iterate over the number of classes
            class_dice = dice_coefficient(y_true[..., i], y_pred[..., i])
            dice_scores.append(class_dice)

        # Convert list to tensor
        dice_scores = K.stack(dice_scores)
            
        # Apply the class weights
        weighted_dice = K.sum(class_weights * (1 - dice_scores))  # 1 - Dice score to convert to loss
        
        return weighted_dice
    
    return wdl


def train_unet_model(model, train_images, train_labels, 
                     val_images, val_labels, 
                     batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=learning_rate,
                     model_save_path="unet_model.h5", class_weights=[1, 1, 1, 1, 1, 1]):
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

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5,  # Reduce by 50%
                                                 patience=3,  # Wait for 5 epochs before reducing
                                                 min_lr=1e-7)  # Minimum learning rate
    

    loss = weighted_categorical_crossentropy(class_weights)
    print("weighted_cc_loss")
    #loss = weighted_dice_loss(class_weights)
    #print("weighted_dice_loss")
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)


    # Compile the model 
    model.compile(optimizer=optimizer, loss=loss, metrics=[dice_coefficient])

    # Print the model summary to verify the architecture
    model.summary()

    # Train the model
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint, early_stopping, reduce_lr])
    
    plot_training(history)

    #wandb.finish()

    print("LR:", learning_rate, "/nEPOCHS:", EPOCHS, "/nBATCHSIZE:", BATCH_SIZE)

    return history



def plot_training(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(history.epoch, loss, 'r', label='Training loss')
    plt.plot(history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('training_loss_drop0.2.png')