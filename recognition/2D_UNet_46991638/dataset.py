import numpy as np
import nibabel as nib
from tqdm import tqdm
import glob
from matplotlib import pyplot
from matplotlib import image
import tensorflow as tf 
import skimage.transform as sk

def to_channels(arr:np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (6,), dtype = dtype)
    for c in channels:
        c = int(c)
        res [... , c : c +1][arr == c] = 1
    return res
# load medical image functions
def load_data_2D (imageNames, normImage = False, categorical = False, dtype = np.float32,
    getAffines = False, early_stop = False):
    '''
    Load medical image data from names , cases list provided into a list for each .

    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory usage .

    normImage : bool(normalise the image 0.0 - 1.0)
    early_stop : Stop loading pre-maturely , leaves arrays mostly empty , for quick loading and testing scripts .
    '''
    affines = []
    #count = 0
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if categorical:
        #neareat-neigboor interpolation
        first_case = sk.resize(first_case, (256,256), order=0, anti_aliasing=False, preserve_range=True)
    else:
        #bi-linear
        first_case = sk.resize(first_case, (256, 256), order=1, anti_aliasing=True, preserve_range=True)
    if len(first_case.shape) == 3:
        first_case = first_case [: ,: ,0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)
    for i, inName in enumerate (tqdm(imageNames)):   
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching ="unchanged") # read disk only

        if categorical:
            #neareat-neigboor interpolation
            inImage = sk.resize(inImage, (256,256), order=0, anti_aliasing=False, preserve_range=True)
        else:
            #bi-linear
            inImage = sk.resize(inImage, (256, 256), order=1, anti_aliasing=True, preserve_range=True)

        affine = niftiImage.affine
        if len (inImage.shape ) == 3:
            inImage = inImage[: ,: ,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i ,: ,: ,:] = inImage
        else: 
            images[i ,: ,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break
    print("H")
    if getAffines:
        return images, affines
    else:
        return images
    

def load(path, label=False):
    image_list = []
    for filename in glob.glob(path + '/*.nii.gz'): 
        image_list.append(filename)
    train_set = load_data_2D(image_list, normImage=False, categorical=label)
    return train_set

def process_data(data_set):
  # the method normalizes training images and adds 4th dimention 

  train_set = data_set
  train_set = (train_set - np.mean(train_set))/ np.std(train_set)
  train_set= (train_set- np.amin(train_set))/ np.amax(train_set- np.amin(train_set))
  train_set = train_set[...,np.newaxis]
  return train_set

class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return int(np.floor(len(self.x) / self.batch_size))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y