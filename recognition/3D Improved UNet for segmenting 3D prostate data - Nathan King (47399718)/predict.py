"""
This script executes the loading and testing process of the trained 3D Improved UNet Model. Additionally, examples showing the
input, true segmentation and predicted segmentation are visualised and saved to depict the performance of the model.

@author Nathan King
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataset import DOWNSIZE_FACTOR, load_mri_data
from train import SAVED_RESULTS_PATH, DATA_PATH, BATCH_LENGTH, BUFFER_SIZE, multiclass_dice_coefficient, background_dsc, body_dsc, bone_dsc, bladder_dsc, rectum_dsc, prostate_dsc

def display(display_list, display_index):
    """
    Display and save animated 3D input, true mask and predicted mask data.
    input: display_list - input, true mask and predicted mask data
           display_index - index used for saving animations
    """
    
    title = ["Input Image", "True Mask", "Predicted Mask"]
    
    fig, ax = plt.subplots(1, 1)
    
    #Frames
    ims = []
    
    #Create each frame
    for i in range(256 // DOWNSIZE_FACTOR):
        
        #Create frame
        im = ax.imshow(tf.keras.utils.array_to_img(display_list[0][i]), animated = True)
        
        if i == 0:
            
            ax.imshow(tf.keras.utils.array_to_img(display_list[0][i]))
        
        #Append frame
        ims.append([im])
    
    #Create animation
    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True,
                                    repeat_delay = 1000)
    
    #Save and show animation
    ani.save(SAVED_RESULTS_PATH + "Input" + str(display_index) + ".gif")
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    
    #Frames
    ims = []
    
    #Create each frame
    for i in range(256 // DOWNSIZE_FACTOR):
        
        #Create frame
        im = ax.imshow(display_list[1][i], vmin = 0, vmax = 5, animated = True)
        
        if i == 0:
            
            ax.imshow(display_list[1][i], vmin = 0, vmax = 5)
            
        #Append frame
        ims.append([im])
        
    #Create animation
    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True,
                                    repeat_delay = 1000)
    
    #Save and show animation
    ani.save(SAVED_RESULTS_PATH + "True" + str(display_index) + ".gif")
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    
    #Frames
    ims = []
    
    #Create each frame
    for i in range(256 // DOWNSIZE_FACTOR):
        
        #Create frame
        im = ax.imshow(display_list[2][i], vmin = 0, vmax = 5, animated = True)
        
        if i == 0:
            
            ax.imshow(display_list[2][i], vmin = 0, vmax = 5)
        
        #Append frame
        ims.append([im])
    
    #Create animation
    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True,
                                    repeat_delay = 1000)
    
    #Save and show animation
    ani.save(SAVED_RESULTS_PATH + "Predicted" + str(display_index) + ".gif")
    plt.show()

def display_and_save_examples(dataset, number_of_examples, model):
    """
    Display examples with accuracy and dice similarity coefficient from a dataset and save them.
    input: dataset - dataset to take from
           number_of_examples - number of examples to take
           model - trained model
    """
    
    prediction_index = 0
    
    #Iterate through each image-segmentation pair
    for image, segmentation in dataset.take(number_of_examples):
        
        #Calculate true segmentation
        true = segmentation
        
        #Calculate predicted segmentation
        predicted = model.predict(image)
        
        #Print statistics
        print("Multiclass DSC:", multiclass_dice_coefficient(true, predicted).numpy())
        print("Background DSC:", background_dsc(true, predicted).numpy())
        print("Body DSC:", body_dsc(true, predicted).numpy())
        print("Bone DSC:", bone_dsc(true, predicted).numpy())
        print("Bladder DSC:", bladder_dsc(true, predicted).numpy())
        print("Rectum DSC:", rectum_dsc(true, predicted).numpy())
        print("Prostate DSC:", prostate_dsc(true, predicted).numpy())
        
        #Process image for displaying
        image_for_display = image[0]
        
        #Process true segmentation for displaying
        true_for_display = [[[[list(x).index(1)] for x in y] for y in z] for z in segmentation[0].numpy()]
        
        #Process predicted segmentation for displaying
        predicted_for_display = model.predict(image)
        predicted_for_display = tf.argmax(predicted_for_display, axis = -1)
        predicted_for_display = predicted_for_display[..., tf.newaxis][0]
        
        #Display and save example
        display([image_for_display, true_for_display, predicted_for_display], prediction_index)
        
        prediction_index += 1

def predict_model():
    """
    Make predictions using the trained model.
    """
    
    #Load and batch test data
    test_dataset = load_mri_data(DATA_PATH, True)[1]
    test_batches = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_LENGTH)

    #Load trained model
    trained_improved_3d_unet_model = keras.models.load_model(SAVED_RESULTS_PATH + "improved_3d_unet_model.keras", custom_objects={"multiclass_dice_coefficient": multiclass_dice_coefficient, "background_dsc": background_dsc, "body_dsc": body_dsc, "bone_dsc": bone_dsc, "bladder_dsc": bladder_dsc, "rectum_dsc": rectum_dsc, "prostate_dsc": prostate_dsc})
    
    #Evaluate the trained model on the test set by calculating the dice similarity
    # coefficients for each class and the multiclass dice similarity coefficient
    trained_improved_3d_unet_model.evaluate(test_batches)
    
    #Display and save examples
    display_and_save_examples(test_batches, 3, trained_improved_3d_unet_model)

if __name__ == "__main__":
    
    predict_model()