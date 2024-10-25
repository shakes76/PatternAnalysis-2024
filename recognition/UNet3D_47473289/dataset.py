# Data loader for leading and preprocessing data.
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import nibabel as nib
import numpy as np
import tqdm as tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import torch.nn as nn
import os
import torchvision.transforms as transforms
import glob



def to_channels ( arr : np . ndarray , dtype = np . uint8 ) -> np . ndarray :
    channels = np . unique ( arr )
    res = np . zeros ( arr . shape + ( len ( channels ) ,) , dtype = dtype )
    for c in channels :
        c = int ( c )
        res [... , c : c +1][ arr == c ] = 1

    return res

transform = transforms.Compose([
    transforms.Resize((128, 128)),
   # transforms.Normalize(0.5, 0.5),
])
l_transform = transforms.Compose([
    transforms.Resize((128, 128)),
])


def load_data_3D(imageNames, normImage=False , categorical=False, dtype=np.float32, 
                    getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names , cases list provided into a list for each .

    This function pre - allocates 5 D arrays for conv3d to avoid excessive memory ↘
    usage .

    normImage : bool ( normalise the image 0.0 -1.0)
    orient : Apply orientation and resample image ? Good for images with large slice ↘
    thickness or anisotropic resolution
    dtype : Type of the data . If dtype = np . uint8 , it is assumed that the data is ↘
    labels
    early_stop : Stop loading pre - maturely ? Leaves arrays mostly empty , for quick ↘
    loading and testing scripts .
    '''
    affines = []


    # ~ interp = ' continuous '
    interp = 'linear'
    if dtype == np.uint8: # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)

        # ~ testResultName = "oriented.nii.gz"
        # ~ niftiImage.to_filename(testResultName)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case [:,:,:,0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels) , dtype=dtype)

    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)


    for i ,inName in enumerate(tqdm.tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if len(first_case.shape) == 3:
            first_case = np.expand_dims(first_case, axis=0)

        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        
        if len (inImage.shape) == 4:
            inImage = inImage [:,:,:,0] # sometimes extra dims in HipMRI_study data

        inImage = inImage [:,:,:depth] # clip slices
        inImage = inImage.astype(dtype)
        
        if normImage:
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage . mean () ) / inImage . std ()
        if categorical :
            inImage = to_channels(inImage, dtype=dtype)
            # ~ images [i,:,:,:,:] = inImage
            images [i,:inImage.shape[0],:inImage.shape[1],:inImage.shape [2],:inImage.shape[3]] = inImage # with pad


      
        else :
            # ~ images [i,:,:,:] = inImage

            images [i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2]] = inImage # with pad
            
        affines.append(affine)
        if i > 20 and early_stop:
            print("STOPPED EARly")
            break

    if getAffines:
        return images, affines
    else:
        print("Returned images")
        return images

class MyCustomDataset(Dataset):
    def __init__(self):
        # load all nii handle in a list
        #self.image_paths = glob.glob(f'{"/Users/charl/Documents/3710Report/PatternAnalysis-2024/recognition/semantic_MRs_anon"}/**/*.nii.gz', recursive=True)
        #self.label_paths = glob.glob(f'{"/Users/charl/Documents/3710Report/PatternAnalysis-2024/recognition/semantic_labels_only"}/**/*.nii.gz', recursive=True)
        self.label_paths = glob.glob(f'{"/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"}/**/*.nii.gz', recursive=True)
        self.image_paths = glob.glob(f'{"/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"}/**/*.nii.gz', recursive=True)
        print(len(self.image_paths))
        print(len(self.label_paths))
  
 
        self.resize = transforms.Compose([transforms.Resize((256, 128))])
        self.up = torch.nn.Upsample(size=(128,128,128))

        self.classes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = load_data_3D([self.image_paths[i]],early_stop=True)
        label = load_data_3D([self.label_paths[i]],early_stop=True)
        print(image.shape)
        image = torch.tensor(image).float().unsqueeze(1)
        label = torch.tensor([self.classes.get(cla.item(), 0) for cla in label.flatten()]).reshape(label.shape)
        label = nn.functional.one_hot(label.squeeze(0), num_classes=6).permute(3,1,0,2).float()
        label = label.unsqueeze(0)
        image = self.up(image).squeeze(0)
        label = self.up(label).squeeze(0)
        print(image.shape)
        print(label.shape)

        return image, label



dataset = MyCustomDataset()
print(dataset[10])
print(dataset[10][1].shape)