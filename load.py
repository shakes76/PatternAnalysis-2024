import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, 
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.
    
    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0]  # sometimes extra dims, remove
    
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
            inImage = inImage[:,:,0]  # sometimes extra dims in HipMRI_study data
        
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, 
                 getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.
    
    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    interp = 'linear'
    
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'
    
    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
    
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:,:,:,0]  # sometimes extra dims, remove
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 4:
            inImage = inImage[:,:,:,0]  # sometimes extra dims in HipMRI_study data
            inImage = inImage[:,:,:depth]  # clip slices
        
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2],:inImage.shape[3]] = inImage  # with pad
        else:
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2]] = inImage  # with pad
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images

def get_dataloaders(semantic_labels_dir, semantic_mrs_dir):
    """Load the dataset and returns the data loaders."""
    semantic_label_files = [os.path.join(semantic_labels_dir, f) for f in os.listdir(semantic_labels_dir) if f.endswith('.gz')]
    semantic_labels, semantic_labels_affines = load_data_3D(semantic_label_files, normImage=False, categorical=True, dtype=np.uint8, getAffines=True)

    semantic_mr_files = [os.path.join(semantic_mrs_dir, f) for f in os.listdir(semantic_mrs_dir) if f.endswith('.gz')]
    semantic_mrs, semantic_mrs_affines = load_data_2D(semantic_mr_files, normImage=False, categorical=False, dtype=np.float32, getAffines=True)

    return semantic_labels, semantic_mrs, semantic_labels_affines, semantic_mrs_affines

def show_batch(semantic_labels, semantic_mrs, filename):
    """Plot images grid of single batch."""
    # Combine label and MR images
    combined = np.concatenate([semantic_labels, semantic_mrs], axis=-1)
    img = make_grid(torch.from_numpy(combined))
    show_images(img, filename)

def show_images(img, filename):
    """Plot images grid of single batch."""
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.savefig(filename)
    plt.clf()

# # Load semantic label files
# semantic_labels_dir = '/Users/ella/Documents/UQ/BM_BCs/Y4S2/COMP3710/report/PatternAnalysis-2024/HipMRI_study_complete_release_v1/semantic_labels_anon'
# semantic_label_files = [os.path.join(semantic_labels_dir, f) for f in os.listdir(semantic_labels_dir) if f.endswith('.gz')]
# semantic_labels, semantic_labels_affines = load_data_3D(semantic_label_files, normImage=False, categorical=True, dtype=np.uint8, getAffines=True)

# # Load semantic MR files
# semantic_mrs_dir = '/Users/ella/Documents/UQ/BM_BCs/Y4S2/COMP3710/report/PatternAnalysis-2024/HipMRI_study_complete_release_v1/semantic_MRs_anon'
# semantic_mr_files = [os.path.join(semantic_mrs_dir, f) for f in os.listdir(semantic_mrs_dir) if f.endswith('.gz')]
# semantic_mrs, semantic_mrs_affines = load_data_3D(semantic_mr_files, normImage=False, categorical=False, dtype=np.float32, getAffines=True)

# print(f"Loaded {len(semantic_label_files)} semantic label files")
# print(f"Loaded {len(semantic_mr_files)} semantic MR files")

# Example usage
semantic_labels_dir = '/Users/ella/Documents/UQ/BM_BCs/Y4S2/COMP3710/report/PatternAnalysis-2024/HipMRI_study_complete_release_v1/semantic_labels_anon'
semantic_mrs_dir = '/Users/ella/Documents/UQ/BM_BCs/Y4S2/COMP3710/report/PatternAnalysis-2024/HipMRI_study_complete_release_v1/semantic_MRs_anon'

semantic_labels, semantic_mrs, semantic_labels_affines, semantic_mrs_affines = get_dataloaders(semantic_labels_dir, semantic_mrs_dir)

print(f"Loaded {semantic_labels.shape[0]} semantic label files")
print(f"Loaded {semantic_mrs.shape[0]} semantic MR files")

show_batch(semantic_labels[0], semantic_mrs[0], 'batch_example.png')
