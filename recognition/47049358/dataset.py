import os
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

IMAGE_FILE_NAME = os.path.join(os.getcwd(), 'semantic_MRs_anon')
LABEL_FILE_NAME = os.path.join(os.getcwd(), 'semantic_labels_anon')
RANDOM_STATE = 47049358

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        if categorical:
            # inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break

    return (images, affines) if getAffines else images

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    dtype : Type of the data. If dtype=np.uint8, it is assumed that the data has labels.
    early_stop : Stop loading prematurely. Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    first_case = niftiImage.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        
        niftiImage = nib.load(inName)
        
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 4:
            inImage = inImage[ : , : , : , 0]  # sometimes extra dims in HipMRI_study data
        
        inImage = inImage[ : , : , : depth]  # clip slices
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2], : inImage.shape[3]] = inImage  # with pad
        else:
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]] = inImage  # with pad

        affines.append(affine)

        if i > 20 and early_stop:
            break

    return (images, affines) if getAffines else images

def load_images_and_labels(image_file_name, label_file_name, early_stop = False):
    images = load_data_3D(imageNames=image_file_name, early_stop=early_stop)
    labels = load_data_3D(imageNames=label_file_name, categorical=True, dtype=np.uint8, early_stop=early_stop)
    return images, labels

def save_data(images, labels, image_save_path, label_save_path):
    with open(image_save_path, 'wb') as f:
        pickle.dump(images, f)
    with open(label_save_path, 'wb') as f:
        pickle.dump(labels, f)

def load_saved_data(image_save_path, label_save_path):
    with open(image_save_path, 'rb') as f:
        images = pickle.load(f)
    with open(label_save_path, 'rb') as f:
        labels = pickle.load(f)
    return images, labels

"""
    PyTorch Dataset class for loading ISIC melanoma detection dataset.

    Parameters:
    - img_path (str): Path to the directory containing image files.
    - mask_path (str): Path to the directory containing mask files.
    - transform (call): Transform to be applied on the images and masks.

    Methods:
    - __len__(): Returns the total number of images in the dataset.
    - __getitem__(idx): Returns the image and its corresponding mask at the given index `idx`. Resizes image and converts to correct colour channels. 
    """
class Prostate3dDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]
        return img, mask

rawImageNames = os.listdir(IMAGE_FILE_NAME)
rawLabelNames = os.listdir(LABEL_FILE_NAME)

# Split the set into train, validation, and test set (70:15:15 for train:valid:test)
X_train, X_rem, y_train, y_rem = train_test_split(rawImageNames, rawLabelNames, train_size=0.8, random_state=RANDOM_STATE) # Split the data in training and remaining set
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=0.5, random_state=RANDOM_STATE)

X_train = [os.path.join(IMAGE_FILE_NAME, image) for image in X_train]
X_val = [os.path.join(IMAGE_FILE_NAME, image) for image in X_val]
X_test = [os.path.join(IMAGE_FILE_NAME, image) for image in X_test]

y_train = [os.path.join(LABEL_FILE_NAME, label) for label in y_train]
y_val = [os.path.join(LABEL_FILE_NAME, label) for label in y_val]
y_test = [os.path.join(LABEL_FILE_NAME, label) for label in y_test]

X_train, y_train = load_images_and_labels(X_train, y_train, early_stop=True)
X_val, y_val = load_images_and_labels(X_val, y_val, early_stop=True)
X_test, y_test = load_images_and_labels(X_test, y_test, early_stop=True)

X_train = X_train[: ,np.newaxis, :, :, :]
X_val = X_val[:,np.newaxis, :, :, :]
X_test = X_test[:, np.newaxis, :, :, :]