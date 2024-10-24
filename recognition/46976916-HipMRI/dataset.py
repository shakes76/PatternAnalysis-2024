#Reading Nifti files - Code from Appendix B
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import skimage.transform as skTrans

import numpy as np
import nibabel as nib
from tqdm import tqdm

def to_channels ( arr : np . ndarray , dtype = np . uint8 ) -> np . ndarray :
    #channels = np . unique ( arr )
    num_classes = 5
    res = np . zeros ( arr . shape + ( num_classes ,) , dtype = dtype )
    for c in range(num_classes) :
        c = int ( c )
        res [... , c : c +1][ arr == c ] = 1

    return res

# load medical image functions
def load_data_2D ( imageNames , normImage = False , categorical = False , dtype = np . float32 ,
    getAffines = False , early_stop = False ) :
    '''
    Load medical image data from names , cases list provided into a list for each .

    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory ↘
    usage .

    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick ↘
    loading and testing scripts .
    '''
    affines = []
    num_to_load = int(len(imageNames) * 0.3)
    imageNames = imageNames[-num_to_load:]
    # get fixed size
    num = len ( imageNames )
    first_case = nib . load ( imageNames [0]) . get_fdata ( caching = "unchanged")
    if len ( first_case . shape ) == 3:
        first_case = first_case [: ,: ,0] # sometimes extra dims , remove
    if categorical :
        first_case = to_channels ( first_case , dtype = dtype )
        rows , cols , channels = first_case . shape
        images = np . zeros (( num , rows , cols , channels ) , dtype = dtype )
    else :
        rows , cols = first_case . shape
        images = np . zeros (( num , rows , cols ) , dtype = dtype )

    for i , inName in enumerate ( tqdm ( imageNames ) ) :
        niftiImage = nib . load ( inName )
        Pre_inImage = niftiImage . get_fdata ( caching = "unchanged") # read disk only
        affine = niftiImage . affine
        #Now loads as Pre_inImage and the Pre_inImage is resized to always be 256,128
        inImage = skTrans.resize(Pre_inImage, (256,128), order=0, preserve_range=True)

        if len ( inImage . shape ) == 3:
            inImage = inImage [: ,: ,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage . astype ( dtype )
        
        if normImage :
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = ( inImage - inImage . mean () ) / inImage . std ()
        if categorical :
                inImage = to_channels ( inImage , dtype = dtype )
                images [i ,: ,: ,:] = inImage
        else :
            images [i ,: ,:] = inImage

        affines . append ( affine )
        if i > 20 and early_stop :
            break

    if getAffines :
        return images , affines
    else :
        return images


class ProstateCancerDataset(Dataset):
    def __init__(self, image_dir, seg_dir, normImage=True, categorical=False, dtype=np.float32, transform=None):
        # Use load_data_2D to load all the input images and ground truth masks at once
        imageNames = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.images = load_data_2D(imageNames, normImage=normImage, categorical=categorical, dtype=dtype)
        segNames = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
        self.segImages = load_data_2D(segNames, normImage=False, categorical=True, dtype=dtype)
        
        self.transform = transform  # Optional transformations (like resizing or normalization)
    
    def __len__(self):
        return len(self.images)  # Return the number of samples

    def __getitem__(self, idx):
        # Get the input image and ground truth mask for the given index
        image = self.images[idx]
        segImage = self.segImages[idx]

        # Apply any transformations (if any)
        if self.transform is not None:
            #image = self.transform(image)
            #segImage = self.transform(segImage)
            augmentations = self.transform(image=image, mask=segImage)
            image = augmentations["image"]
            segImage = augmentations["mask"]

        return image, segImage
        