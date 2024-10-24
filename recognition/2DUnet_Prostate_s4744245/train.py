import numpy as np
import random
#import wandb

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


# plot the training/validation curves from the history
import matplotlib.pyplot as plt

#define training variables
BATCH_SIZE = 1
EPOCHS = 50
n_classes = 6
learning_rate = 0.005

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
    
    #weightAndBiasCallback = tf.keras.callbacks.LambdaCallback(
    #    on_epoch_end=lambda epoch, logs: weightsBiasDict.update({epoch: model.get_weights()}))

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
    
    loss = weighted_categorical_crossentropy(class_weights)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)


    # Compile the model 
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Print the model summary to verify the architecture
    model.summary()

    # Train the model
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint, early_stopping])
    
    #plot_training(history)

    #wandb.finish()

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
    plt.savefig('training_loss.png')