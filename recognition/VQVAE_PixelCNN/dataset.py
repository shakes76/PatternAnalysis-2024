"""
This module creates the dataset for the project.
""" 

# Importing libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nibabel as nib
from tqdm import tqdm
from collections import Counter
from torchvision.transforms import Normalize, Compose, ToTensor
import torchvision.transforms as transforms
from PIL import Image
from train_VQVAE import BATCH_SIZE

def count_image_dimensions(image_dir):
    """
    Count the distribution of image dimensions in the dataset.
    
    Parameters:
        - image_dir: The directory containing the .nii.gz image files.
    
    Returns:
        - dimension_counter: A Counter object with the distribution of dimensions.
    """
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
    
    dimension_counter = Counter()
    
    for image_file in tqdm(image_files, desc="Counting image dimensions"):
        # Load the Nifti image
        nifti_image = nib.load(image_file)
        # Get the shape of the image data
        image_shape = nifti_image.get_fdata(caching='unchanged').shape
        
        # For 3D images, consider only the first slice to count 2D dimensions
        if len(image_shape) == 3:
            image_shape = image_shape[:2]
        
        # Update the counter with the current image's dimensions
        dimension_counter[image_shape] += 1
    
    # Print the results
    print("Image Dimension Distribution|:")
    for dim, count in dimension_counter.items():
        print(f"Dimension: {dim}, Count: {count}")


def to_channels(arr: np.ndarray, dtype = np.uint8 ) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels), ), dtype = dtype)
    for c in channels:
        c = int(c)
        res[..., c:c +1][arr == c] = 1
    
    return res


# load medical image functions
def load_data_2D(imageNames , normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False):
    '''
    Load medical image data from names , cases list provided into a list for each.
    This function pre - allocates 4D arrays for conv2d to avoid excessive memory usage.
    normImage : bool(normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick loading and testing scripts.
    '''
    
    affines = []
    
    #get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames, desc ="Loading the images")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching ='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np. linalg . norm ( inImage )
            #~ inImage = 255. * inImage / inImage .max ()
            inImage = (inImage - inImage.mean()) /inImage.std()
        if categorical:
            inImage = utils.to_channels ( inImage , dtype = dtype )
            images [i, :, :, :] = inImage
        else :
            images [i, :, :] = inImage
            
        affines.append(affine)
        if i > 20 and early_stop :
            break
            
    if getAffines:
        return images, affines
    else:
        return images
    

def filter_image_files_by_dimension(image_files, target_size=(256, 128)):
    """
    Filter the list of image files by their dimensions.
    
    Parameters:
        - image_files: List of file paths to Nifti images (.nii.gz).
        - target_size: Tuple, the desired image dimensions (default is (256, 128)).
    
    Returns:
        - valid_image_files: List of file paths with matching dimensions.
    """
    valid_image_files = []

    for image_file in tqdm(image_files, desc="Filtering images by dimensions"):
        nifti_image = nib.load(image_file)
        image_shape = nifti_image.get_fdata(caching='unchanged').shape

        # For 3D images, check the 2D slice dimensions
        if len(image_shape) == 3:
            image_shape = image_shape[:2]

        # Only keep images with the desired dimensions
        if image_shape == target_size:
            valid_image_files.append(image_file)
    
    return valid_image_files


def get_all_image_files(root_dir):
    """
    Get all image file paths from all subdirectories within the root directory.

    Parameters:
        - root_dir: The root directory containing multiple folders of images.
    
    Returns:
        - image_files: List of all image file paths across all subdirectories.
    """
    image_files = []

    # Walk through all subdirectories within the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get all .nii.gz files from the current directory
        folder_images = [os.path.join(dirpath, f) for f in filenames if f.endswith('.nii.gz')]
        image_files.extend(folder_images)  # Add to the list of image files

    return image_files

class ProstateMRIDataset(Dataset):
    """
    A custom Dataset class for loading prostate MRI images from Folder1, Folder2, and Folder3.
    Only images with the desired dimensions (e.g., 256x128) will be loaded.
    """
    
    def __init__(self, root_dir, normImage=False, categorical=False, dtype=np.float32, getAffines=False, target_size=(256, 128)):
        """
        Initialize the dataset by loading and filtering all images from Folder1, Folder2, and Folder3.
        
        Parameters:
            - root_dir: The root directory containing Folder1, Folder2, and Folder3.
            - normImage: Boolean, if True, normalizes the images.
            - categorical: Boolean, if True, converts images into one-hot encoding.
            - dtype: Data type for the images (default is np.float32).
            - getAffines: Boolean, if True, also returns affine matrices.
            - target_size: Tuple, the desired image size to keep (default is (256, 128)).
        """
        self.root_dir = root_dir
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.getAffines = getAffines
        self.target_size = target_size

        # Get all image file paths from dataset folder
        all_image_files = get_all_image_files(self.root_dir)
        
        # Filter the image files by their dimensions
        self.image_files = filter_image_files_by_dimension(all_image_files, target_size=self.target_size)

                # Define the transform for resizing, remove ToTensor
        self.transform = transforms.Compose([
            # transforms.Resize(final_size),  # Resize to 128x128
            transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
            ToTensor()
        ])
        
        # Load the valid images using the `load_data_2D` function
        if getAffines:
            self.images, self.affines = load_data_2D(self.image_files, normImage=self.normImage,
                                                     categorical=self.categorical, dtype=self.dtype,
                                                     getAffines=self.getAffines)
        else:
            self.images = load_data_2D(self.image_files, normImage=self.normImage,
                                                     categorical=self.categorical, dtype=self.dtype,
                                                     getAffines=self.getAffines)    

    def __len__(self):
        """Return the total number of valid images."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return an image by index."""
        image = self.images[idx]

        # Convert the NumPy array to PIL Image before resizing
        image = Image.fromarray(image)

        # Apply transformations, including resizing
        image = self.transform(image)  # Apply Resize
        
        if self.categorical:
            image = image.astype(np.float32)  # Ensure one-hot encoded data is in float32 format
        return np.array(image)  # Convert back to NumPy array if needed


def get_dataloader(root_dir, batch_size=BATCH_SIZE, normImage=False, categorical=False, target_size=(256, 128)):
    """
    Create a PyTorch DataLoader for batching and shuffling the dataset from Folder1, Folder2, and Folder3.
    
    Parameters:
        - root_dir: The root directory containing Folder1, Folder2, and Folder3.
        - batch_size: Number of samples per batch (default: 8).
        - normImage: Boolean, if True, normalizes the images.
        - categorical: Boolean, if True, converts images into one-hot encoding.
        - target_size: Tuple, the desired image size to keep (default is (256, 128)).
    
    Returns:
        - DataLoader: A PyTorch DataLoader instance for batching.
    """
    dataset = ProstateMRIDataset(root_dir, normImage=normImage, categorical=categorical, target_size=target_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Dataset Visualizer

def visualize_samples(dataloader, num_samples=4, save_path='Output/sample_visualization.png'):
    """
    Visualize a few sample images from the dataset.
    
    Parameters:
        - dataloader: The DataLoader object containing the dataset.
        - num_samples: The number of samples to visualize.
        - save_path: The path to save the output image.
    """


    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Loading the dataset
    dataset = dataloader.dataset
    
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        sample_image = dataset[i]  # Get a sample image
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_image.squeeze(), cmap='gray')  # Remove channel dimension if needed
        # plt.imshow(sample_image.squeeze())
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.savefig(save_path)  # Save the plot
    plt.show()
    plt.close()  # Close the plot to free up memory
