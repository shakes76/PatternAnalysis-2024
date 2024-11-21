"""
DataLoader, load and preprocess 2D slices of HipMRI Study on Prostate Cancer in Nifti file.

@author Thuan Pham - 46964472
"""
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from keras.utils import Sequence
import skimage.transform
from tqdm import tqdm

# Convert image to categorical one hot encoding channels
def to_channels(arr: np.ndarray, dtype=np.uint8, num_channels=6) -> np.ndarray:
    channels = np.arange(num_channels)
    res = np.zeros(arr.shape + (len(channels), ), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1
    return res

# Load 2D Nifti medical image file
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    affines = []
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] # remove extra dims
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # remove extra dims
        # Transform and resize image if different
        if inImage.shape != (rows, cols): 
            inImage = skimage.transform.resize(inImage, (rows, cols), 
                                              order=1, preserve_range=True)
        inImage = inImage.astype(dtype)
        if normImage: # Normalise data
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical: # One hot encoding
            inImage = np.round(inImage)
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        affines.append(affine)
        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images

# Data generator class to load to model
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# Load data from file to data generator and return
def load_data_tf(image_dir, seg_dir, dtype=np.float32, batch_size=32):
    # List all file name in image dir
    image_list = os.listdir(image_dir)
    image_path = []
    seg_path = []
    # Append full path to image and segmentation files
    for image in image_list:
        image_path.append(os.path.join(image_dir, image))
        seg_path.append(os.path.join(seg_dir, image.replace("case", "seg")))
    # Load Nifti file data
    images = load_data_2D(image_path, normImage=True, dtype=dtype)
    segs = load_data_2D(seg_path, categorical=True, dtype=dtype)
    # Create and load data to DataGenerator class
    dataset = DataGenerator(images, segs, batch_size)
    return dataset

# Load random data to visualise predict
def load_data_predict(image_dir, seg_dir, number_image):
    # List all file name in image dir
    image_list = os.listdir(image_dir)
    image_path = []
    seg_path = []
    # Append full path to image and segmentation files
    for i in range(number_image):
        index = np.random.randint(1, len(image_list))
        image_path.append(os.path.join(image_dir, image_list[index]))
        seg_path.append(os.path.join(seg_dir, image_list[index].replace("case", "seg")))
    # Load Nifti file data
    images = load_data_2D(image_path)
    images_norm = load_data_2D(image_path, normImage=True)
    segs = load_data_2D(seg_path)
    return images, images_norm, segs