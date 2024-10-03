# dataset.py
import numpy as np
from PIL import Image
import tensorflow as tf
import glob

# Load and resize images
def load_and_resize_images(filelist, target_size=(256, 256)):
    images = []
    for filename in filelist:
        # Load image
        img = Image.open(filename)
        # Convert to numpy array
        img_array = np.array(img, dtype="float32")
        # Add channel dimension (assume grayscale images)
        if len(img_array.shape) == 2:  # Check if the image is 2D (grayscale)
            img_array = np.expand_dims(img_array, axis=-1)  # Convert to (height, width, 1)
        # Resize image using tf.image.resize
        img_resized = tf.image.resize(img_array, target_size)
        images.append(img_resized.numpy())
    return np.array(images)

# Data loading function
def load_data():
    # Define file paths
    filelist_ad_train = glob.glob('/home/groups/comp3710/ADNI/AD_NC/train/AD/*.jpeg')
    filelist_cn_train = glob.glob('/home/groups/comp3710/ADNI/AD_NC/train/NC/*.jpeg')
    filelist_ad_test = glob.glob('/home/groups/comp3710/ADNI/AD_NC/test/AD/*.jpeg')
    filelist_cn_test = glob.glob('/home/groups/comp3710/ADNI/AD_NC/test/NC/*.jpeg')
    
    # Load and resize images
    images_train_AD = load_and_resize_images(filelist_ad_train, target_size=(256, 256))
    images_train_CN = load_and_resize_images(filelist_cn_train, target_size=(256, 256))
    images_test_AD = load_and_resize_images(filelist_ad_test, target_size=(256, 256))
    images_test_CN = load_and_resize_images(filelist_cn_test, target_size=(256, 256))
    
    # Concatenate all images into one array
    images = np.concatenate((images_train_AD, images_train_CN, images_test_AD, images_test_CN), axis=0)
    
    # Normalize pixel values from [0, 255] to [-1, 1]
    images = (images - 127.5) / 127.5
    
    # Add an extra channel dimension
    images = images[:, :, :, np.newaxis]
    
    return images
