import os
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader

IMAGE_FILE_NAME = os.path.join(os.getcwd(), 'semantic_MRs_anon')
LABEL_FILE_NAME = os.path.join(os.getcwd(), 'semantic_labels_anon')
BATCH_SIZE = 1
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

# Global constants for the dimensions to which images and masks will be resized
# TRANSFORMED_X = 256  # width after resizing
# TRANSFORMED_Y= 256  # height after resizing

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
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    # def match_mask_to_image(self, img_filename):
    #     base_name = os.path.splitext(img_filename)[0]  # This removes the .jpg or any extension
    #     return base_name + '_segmentation.png'

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.labels[idx]

        # img = Image.open(img_name).convert('RGB')  # Convert to RGB
        # mask = Image.open(mask_name).convert('L')  # Convert to grayscale for segmentation masks.

        # img = img.resize((TRANSFORMED_Y, TRANSFORMED_X))
        # mask = mask.resize((TRANSFORMED_Y, TRANSFORMED_X), Image.NEAREST)

        # if self.transform:
        #     img = self.transform(img)
        #     mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0  # Convert to single channel tensor

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

train_set = Prostate3dDataset(X_train, y_train)
validation_set = Prostate3dDataset(X_val, y_val)
test_set = Prostate3dDataset(X_test, y_test)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
validation_loader = DataLoader(validation_set, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

# train_dir = os.path.join(os.getcwd(), 'train_set')
# val_dir = os.path.join(os.getcwd(), 'validation_set')
# test_dir = os.path.join(os.getcwd(), 'test_set')
# os.makedirs(train_dir, exist_ok = True)
# os.makedirs(test_dir, exist_ok = True)

# train_img_file = os.path.join(train_dir, 'train_images.pkl')
# train_label_file = os.path.join(train_dir, 'train_labels.pkl')

# val_img_file = os.path.join(val_dir, 'validation_images.pkl')
# val_label_file = os.path.join(val_dir, 'validation_labels.pkl')

# test_img_file = os.path.join(test_dir, 'test_images.pkl')
# test_label_file = os.path.join(test_dir, 'test_labels.pkl')

# save_data(images=X_train, labels=y_train, image_save_path=train_img_file, label_save_path=train_label_file)
# save_data(images=X_val, labels=y_val, image_save_path=val_img_file, label_save_path=val_label_file)
# save_data(images=X_test, labels=y_test, image_save_path=test_img_file, label_save_path=test_label_file)
