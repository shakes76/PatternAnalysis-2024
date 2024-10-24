import numpy as np
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate the Structural Similarity Index (SSIM) between two images.
    
    Args:
        original (np.ndarray): the original image.
        reconstructed (np.ndarray): the reconstructed image.
        
    Returns:
        float: the SSIM between the two images.
    """
    ssim_value = ssim(original, reconstructed, data_range=original.max() - original.min())
    return ssim_value


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file and return the data as a dictionary.
    
    Args:
        file_path (str): path to the YAML file.
        
    Returns:
        dict: the data from the YAML file.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except:
            raise Exception("Invalid YAML file")


def combine_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Combine two images side by side.
    
    Args:
        image1 (np.ndarray): the first image.
        image2 (np.ndarray): the second image.
        
    Returns:
        np.ndarray: the combined image.
    """
    image1 = Image.fromarray(image1)
    image1 = image1.resize((128, 256))
    image1 = np.array(image1)
    
    image2 = Image.fromarray(image2)
    image2 = image2.resize((128, 256))
    image2 = np.array(image2)

    image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
    return np.concatenate((image1, image2), axis=1)
