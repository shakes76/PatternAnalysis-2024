import numpy as np
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