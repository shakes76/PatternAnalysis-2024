"""
This script executes the training, validating, testing and saving process of the 3D Improved UNet Model. Additionally,
accuracy, loss, multiclass dice coefficient and dice similarity coefficient plots are created and saved to visualise
the performance of the model.

@author Nathan King
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt

from modules import improved_3d_unet_model
from dataset import DOWNSIZE_FACTOR, load_mri_data

#Paths for saved results and MRI data
SAVED_RESULTS_PATH = "C:/Users/nk200/Downloads/" #This is my path, please change when using
DATA_PATH = "C:/Users/nk200/Downloads/HipMRI_study_complete_release_v1/" #This is my path, please change when using

BATCH_LENGTH = 2
BUFFER_SIZE = 64

EPOCHS = 10

def multiclass_dice_coefficient(y_true, y_pred):
    """
    Calculates the multiclass dice similarity coefficient between y_true and y_pred using the formula defined in the reference.
    Reference [https://arxiv.org/pdf/1802.10508v1]
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: mdsc - multiclass dice similarity coefficient between y_true and y_pred
    """
    
    #Convert y_pred to one hot encoding
    y_pred = tf.one_hot(tf.math.argmax(y_pred, 4), 6)
    
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    #Calculate size of intersection for each label
    y_pred_true = tf.math.multiply(y_true, y_pred)
    y_pred_true_sum = tf.reduce_sum(y_pred_true, [0, 1, 2, 3])
    
    #Calculate size of true values for each label
    y_true_sum = tf.reduce_sum(y_true, [0, 1, 2, 3])
    
    #Calculate size of predicted values for each label
    y_pred_sum = tf.reduce_sum(y_pred, [0, 1, 2, 3])
    
    #Smoothing constant
    smooth = tf.constant([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001])
    
    #Multiclass dice similarity coefficient
    mdsc = (2 / 6) * tf.reduce_sum(tf.math.divide(tf.math.add(y_pred_true_sum, smooth), tf.math.add(tf.math.add(y_true_sum, y_pred_sum), smooth)), 0)
    
    return mdsc

def dice_coefficient(y_true, y_pred, class_number):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the class specificed by class_number.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
           class_number - class to calculate dice similarity coefficient for
    output: dsc - dice similarity coefficient between y_true and y_pred for the class specificed by class_number
    """
    
    #Convert y_pred to one hot encoding
    y_pred = tf.one_hot(tf.math.argmax(y_pred, 4), 6)
    
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    #Calculate size of intersection for each label
    y_pred_true = tf.math.multiply(y_true, y_pred)
    y_pred_true_sum = tf.reduce_sum(y_pred_true, [0, 1, 2, 3])
    
    #Calculate size of true values for each label
    y_true_sum = tf.reduce_sum(y_true, [0, 1, 2, 3])
    
    #Calculate size of predicted values for each label
    y_pred_sum = tf.reduce_sum(y_pred, [0, 1, 2, 3])
    
    #Smoothing constant
    smooth = tf.constant([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001])
    
    #Dice similarity coefficient
    dsc = 2 * tf.math.divide(tf.math.add(y_pred_true_sum, smooth), tf.math.add(tf.math.add(y_true_sum, y_pred_sum), smooth))[class_number]
    
    return dsc

def background_dsc(y_true, y_pred):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the background class.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: bdsc - dice similarity coefficient between y_true and y_pred for the background class
    """
    
    bdsc = dice_coefficient(y_true, y_pred, 0)
    
    return bdsc

def body_dsc(y_true, y_pred):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the body class.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: bdsc - dice similarity coefficient between y_true and y_pred for the body class
    """
    
    bdsc = dice_coefficient(y_true, y_pred, 1)
    
    return bdsc

def bone_dsc(y_true, y_pred):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the bone class.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: bdsc - dice similarity coefficient between y_true and y_pred for the bone class
    """
    
    bdsc = dice_coefficient(y_true, y_pred, 2)
    
    return bdsc

def bladder_dsc(y_true, y_pred):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the bladder class.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: bdsc - dice similarity coefficient between y_true and y_pred for the bladder class
    """
    
    bdsc = dice_coefficient(y_true, y_pred, 3)
    
    return bdsc

def rectum_dsc(y_true, y_pred):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the rectum class.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: rdsc - dice similarity coefficient between y_true and y_pred for the rectum class
    """
    
    rdsc = dice_coefficient(y_true, y_pred, 4)
    
    return rdsc

def prostate_dsc(y_true, y_pred):
    """
    Calculates the dice similarity coefficient between y_true and y_pred for the prostate class.
    input: y_true - true segmentation data
           y_pred - predicted segmentation data
    output: pdsc - dice similarity coefficient between y_true and y_pred for the prostate class
    """
    
    pdsc = dice_coefficient(y_true, y_pred, 5)
    
    return pdsc

def train_model():
    """
    Train the model and calculate training, validation and test results.
    """
    
    #Load and batch data
    train_dataset, test_dataset, validate_dataset = load_mri_data(DATA_PATH, False)
    train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_LENGTH).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = validate_dataset.batch(BATCH_LENGTH)
    test_batches = test_dataset.batch(BATCH_LENGTH)

    #Build model and print summary of the model
    mri_improved_3d_unet_model = improved_3d_unet_model()
    tf.config.list_physical_devices("GPU")
    mri_improved_3d_unet_model.summary()

    #Compile the model so it calulates dice similarity coefficients during training
    mri_improved_3d_unet_model.compile(optimizer = tf.keras.optimizers.Adam(),
                       loss = "categorical_crossentropy",
                       metrics = ["accuracy", multiclass_dice_coefficient, background_dsc, body_dsc, bone_dsc, bladder_dsc, rectum_dsc, prostate_dsc])

    EPOCH_STEPS = len(list(train_dataset)) // BATCH_LENGTH
    VALIDATION_STEPS = (len(list(validate_dataset)) // BATCH_LENGTH) // 5
    
    #Train model
    history = mri_improved_3d_unet_model.fit(train_batches,
                                   epochs = EPOCHS,
                                   steps_per_epoch = EPOCH_STEPS,
                                   validation_steps = VALIDATION_STEPS,
                                   validation_data = validation_batches,
                                   verbose = True)

    #Plot and save the training and validation accuracy of the model for each epoch
    plt.plot(range(EPOCHS), history.history["accuracy"], label = "Training Accuracy")
    plt.plot(range(EPOCHS), history.history["val_accuracy"], label = "Validation Accuracy")
    plt.legend(loc="upper left")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(SAVED_RESULTS_PATH + "Accuracy.png")
    plt.show()
    
    #Plot and save the training and validation loss of the model for each epoch
    plt.plot(range(EPOCHS), history.history["loss"], label = "Training Loss")
    plt.plot(range(EPOCHS), history.history["val_loss"], label = "Validation Loss")
    plt.legend(loc="upper left")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(SAVED_RESULTS_PATH + "Loss.png")
    plt.show()
    
    #Plot and save the training and validation multiclass dice coefficient of the model for each epoch
    plt.plot(range(EPOCHS), history.history["multiclass_dice_coefficient"], label = "Training Multiclass Dice Coefficient")
    plt.plot(range(EPOCHS), history.history["val_multiclass_dice_coefficient"], label = "Validation Multiclass Dice Coefficient")
    plt.legend(loc="upper left")
    plt.title("Multiclass Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Multiclass Dice Coefficient")
    plt.savefig(SAVED_RESULTS_PATH + "MulticlassDiceCoefficient.png")
    plt.show()
    
    #Plot and save the training dice similarity coefficients (for each class) of the model for each epoch
    plt.plot(range(EPOCHS), history.history["background_dsc"], label = "Background DSC")
    plt.plot(range(EPOCHS), history.history["body_dsc"], label = "Body DSC")
    plt.plot(range(EPOCHS), history.history["bone_dsc"], label = "Bone DSC")
    plt.plot(range(EPOCHS), history.history["bladder_dsc"], label = "Bladder DSC")
    plt.plot(range(EPOCHS), history.history["rectum_dsc"], label = "Rectum DSC")
    plt.plot(range(EPOCHS), history.history["prostate_dsc"], label = "Prostate DSC")
    plt.legend(loc="upper left")
    plt.title("Training Dice Similarity Coefficients For Each Class")
    plt.xlabel("Epoch")
    plt.ylabel("Training Dice Similarity Coefficient")
    plt.savefig(SAVED_RESULTS_PATH + "TrainingDiceSimilarityCoefficients.png")
    plt.show()
    
    #Plot and save the validation dice similarity coefficients (for each class) of the model for each epoch
    plt.plot(range(EPOCHS), history.history["val_background_dsc"], label = "Background DSC")
    plt.plot(range(EPOCHS), history.history["val_body_dsc"], label = "Body DSC")
    plt.plot(range(EPOCHS), history.history["val_bone_dsc"], label = "Bone DSC")
    plt.plot(range(EPOCHS), history.history["val_bladder_dsc"], label = "Bladder DSC")
    plt.plot(range(EPOCHS), history.history["val_rectum_dsc"], label = "Rectum DSC")
    plt.plot(range(EPOCHS), history.history["val_prostate_dsc"], label = "Prostate DSC")
    plt.legend(loc="upper left")
    plt.title("Validation Dice Similarity Coefficients For Each Class")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice Similarity Coefficient")
    plt.savefig(SAVED_RESULTS_PATH + "ValidationDiceSimilarityCoefficients.png")
    plt.show()

    #Evaluate the model on the test set by calculating the dice similarity
    # coefficients for each class and the multiclass dice similarity coefficient
    mri_improved_3d_unet_model.evaluate(test_batches)

    #Save the trained model to use for predictions
    mri_improved_3d_unet_model.save(SAVED_RESULTS_PATH + "improved_3d_unet_model.keras")

if __name__ == "__main__":
    
    train_model()