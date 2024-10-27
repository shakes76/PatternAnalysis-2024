import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from dataset import load_data
from modules import unet_model
from util import BATCH_SIZE, learning_rate, run, n_classes, EPOCHS, dice_coefficient

import matplotlib.pyplot as plt


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


#check if GPU is available
tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

images_train, images_test, images_validate, images_seg_test, images_seg_train, images_seg_validate = load_data()


from sklearn.utils import class_weight

labels_train = np.argmax(images_seg_train, axis=-1)
labels_train = labels_train.flatten()
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)

print(class_weights)

# Initialize the U-Net model
model = unet_model(n_classes, dropout_rate=0.2, input_size=(256, 128, 1))

# Train the U-Net model
history = train_unet_model(model, images_train, images_seg_train, 
                           images_validate, images_seg_validate, 
                           model_save_path=f"best_unet_model_{run}.h5",
                           class_weights=class_weights) 