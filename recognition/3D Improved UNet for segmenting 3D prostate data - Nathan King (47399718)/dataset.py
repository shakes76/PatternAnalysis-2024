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

data_path = "C:/Users/nk200/Downloads/HipMRI_study_complete_release_v1/"

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
    
    filename = os.fsdecode(file)
        
    #Get MRI data
    img = load_data_3D([data_path + "non_seg/" + filename + "/" + filename])[0]
    img = img.tolist()
    
    data = [[[[x/255] for x in y[:128]] for y in z[:256]] for z in img[:256]]
    
    #Get labelled data
    img_seg = load_data_3D([data_path + "seg/" + filename[:-8] + "SEMANTIC_" + filename[-8:] + "/" + filename[:-8] + "SEMANTIC_" + filename[-8:]])[0]
    img_seg = img_seg.tolist()
        
    #Convert to one-hot encoding
    data_seg = [[[((int(x) * [0]) + [1] + ((5 - int(x)) * [0])) for x in y[:128]] for y in z[:256]] for z in img_seg[:256]]
    
    #Split data into training, testing and validation sets
    if file_index < 0.8 * number_of_files:
        
        x_train.append((data, data_seg))
        
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