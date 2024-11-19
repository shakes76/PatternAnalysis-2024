"""
This script contains a function used to downscale and load the training,
testing and validation data from the MRI and label .nii files.

@author Nathan King
"""

import tensorflow as tf
import numpy as np
import os
import nibabel as nib
from tqdm import tqdm

def load_data_3D(imageNames , normImage = False , categorical = False , dtype = np.float32 ,
        getAffines = False , orient = False , early_stop = False) :
    """
    Load medical image data from names , cases list provided into a list for each .
    This function pre - allocates 5D arrays for conv3d to avoid excessive memory &
    usage .
    normImage : bool(normalise the image 0.0 -1.0)
    orient : Apply orientation and resample image ? Good for images with large slice &
    thickness or anisotropic resolution
    dtype : Type of the data.If dtype =np.uint8 , it is assumed that the data is &
    labels
    early_stop : Stop loading pre - maturely? Leaves arrays mostly empty , for quick &
    loading and testing scripts .
    Reference [Project Specification Sheet]
    """
    affines = []
    #~ interp = " continuous "
    interp = "linear "
    if dtype == np.uint8 : # assume labels
        interp = "nearest "

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames [0])
    if orient :
        niftiImage = im.applyOrientation(niftiImage , interpolation = interp , scale =1)
    first_case = niftiImage.get_fdata(caching = "unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case [: ,: ,: ,0] # sometimes extra dims , remove
    if categorical :
        first_case = to_channels(first_case , dtype = dtype)
        rows , cols , depth , channels = first_case.shape
        images = np.zeros ((num , rows , cols , depth , channels) , dtype = dtype)
    else :
        rows , cols , depth = first_case.shape
        images = np.zeros ((num , rows , cols , depth) , dtype = dtype)
    for i , inName in enumerate(tqdm(imageNames)) :
        niftiImage = nib.load(inName)
        if orient :
            niftiImage = im.applyOrientation(niftiImage , interpolation = interp ,
                scale =1)
        inImage = niftiImage.get_fdata(caching = "unchanged") # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage [: ,: ,: ,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage [: ,: ,: depth ] # clip slices
        inImage = inImage.astype(dtype)
        if normImage :
            #~ inImage = inImage / np. linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max ()
            inImage =(inImage - inImage.mean ()) / inImage.std ()
        if categorical :
            inImage = utils.to_channels(inImage , dtype = dtype)
            #~ images [i ,: ,: ,: ,:] = inImage
            images [i ,: inImage.shape [0] ,: inImage.shape [1] ,: inImage.shape [2] ,: inImage
                .shape [3]] = inImage # with pad
        else :
            #~ images [i ,: ,: ,:] = inImage
            images [i ,: inImage.shape [0] ,: inImage.shape [1] ,: inImage.shape [2]] = inImage # with pad

        affines.append(affine)
        if i > 20 and early_stop :
            break

    if getAffines :
        return images , affines
    else :
        return images

#Factor to downscale the data
DOWNSIZE_FACTOR = 4

#Augmentation using flips in all 3 dimensions
FLIP_AUGMENTERS = ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1))

def load_mri_data(data_path, only_testing):
    """
    Load all the data, downscale it, augment it, and split it into training, testing and validation sets.
    input: data_path - location of the data
           only_testing - whether only testing data is required
    output: (train_dataset, test_dataset, validate_dataset) - preprocessed data
    """
    
    x_train = []
    x_test = []
    x_validate = []
    
    #Get directory
    directory = os.fsencode(data_path + "non_seg")
    file_index = -1
    number_of_files = len(os.listdir(directory))
    print("Number of files:", number_of_files)
    
    #Iterate through each MRI with its corresponing label
    for file in os.listdir(directory):
        
        file_index += 1
        
        if file_index == number_of_files:
            
            break
        
        #For only loading testing data
        if only_testing == True and file_index < 0.9 * number_of_files:
            
            continue
        
        filename = os.fsdecode(file)
        
        #Get MRI data
        img = load_data_3D([data_path + "non_seg/" + filename + "/" + filename])[0]
        img = img.tolist()
        
        #Downscale
        data = [[[[x/255] for x in y[:128:DOWNSIZE_FACTOR]] for y in z[:256:DOWNSIZE_FACTOR]] for z in img[:256:DOWNSIZE_FACTOR]]
        
        #Get labelled data
        img_seg = load_data_3D([data_path + "seg/" + filename[:-8] + "SEMANTIC_" + filename[-8:] + "/" + filename[:-8] + "SEMANTIC_" + filename[-8:]])[0]
        img_seg = img_seg.tolist()
        
        #Downscale and convert to one-hot encoding
        data_seg = [[[((int(x) * [0]) + [1] + ((5 - int(x)) * [0])) for x in y[:128:DOWNSIZE_FACTOR]] for y in z[:256:DOWNSIZE_FACTOR]] for z in img_seg[:256:DOWNSIZE_FACTOR]]
        
        #Split data into training, testing and validation sets
        if file_index < 0.8 * number_of_files:
            
            #Create 8 copies of data and data_seg using all combinations of reflections in 3 dimensions
            flipped_data = [[[y[::flip[2]] for y in z[::flip[1]]] for z in data[::flip[0]]] for flip in FLIP_AUGMENTERS]
            flipped_data_seg = [[[y[::flip[2]] for y in z[::flip[1]]] for z in data_seg[::flip[0]]] for flip in FLIP_AUGMENTERS]
            for i in range(8):
                
                x_train.append((flipped_data[i], flipped_data_seg[i]))
            
        elif file_index < 0.9 * number_of_files:
            
            x_validate.append((data, data_seg))
            
        else:
            
            x_test.append((data, data_seg))
        
    #Generator for training dataset
    def load_train():
        
        for data in x_train:
            
            yield data[0], data[1]
    
    #Generator for testing dataset
    def load_test():
        
        for data in x_test:
            
            yield data[0], data[1]
            
    #Generator for validation dataset
    def load_validate():
        
        for data in x_validate:
            
            yield data[0], data[1]
    
    #Load training data into dataset
    train_dataset = tf.data.Dataset.from_generator(
        load_train,
        output_signature=(
            tf.TensorSpec(shape=(256 // DOWNSIZE_FACTOR, 256 // DOWNSIZE_FACTOR, 128 // DOWNSIZE_FACTOR, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256 // DOWNSIZE_FACTOR, 256 // DOWNSIZE_FACTOR, 128 // DOWNSIZE_FACTOR, 6), dtype=tf.int32)))
    
    #Load testing data into dataset
    test_dataset = tf.data.Dataset.from_generator(
        load_test,
        output_signature=(
            tf.TensorSpec(shape=(256 // DOWNSIZE_FACTOR, 256 // DOWNSIZE_FACTOR, 128 // DOWNSIZE_FACTOR, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256 // DOWNSIZE_FACTOR, 256 // DOWNSIZE_FACTOR, 128 // DOWNSIZE_FACTOR, 6), dtype=tf.int32)))
    
    #Load validation data into dataset
    validate_dataset = tf.data.Dataset.from_generator(
        load_validate,
        output_signature=(
            tf.TensorSpec(shape=(256 // DOWNSIZE_FACTOR, 256 // DOWNSIZE_FACTOR, 128 // DOWNSIZE_FACTOR, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256 // DOWNSIZE_FACTOR, 256 // DOWNSIZE_FACTOR, 128 // DOWNSIZE_FACTOR, 6), dtype=tf.int32)))
    
    return (train_dataset, test_dataset, validate_dataset)