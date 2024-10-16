import numpy as np
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.transforms import Compose


def calculate_ssim(original, reconstructed):
    ssim_value = ssim(original, reconstructed, data_range=original.max() - original.min())
    return ssim_value


def read_yaml_file(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except:
            raise Exception("Invalid YAML file")
        

def get_transforms(transform_config: list) -> Compose:
    transform_list = []
    for transform in transform_config:
        transform_name = transform.get('name')
        transform_params = transform.get('params', {})
        transform_fn = getattr(transforms, transform_name)
        transform_list.append(transform_fn(**transform_params))
    return Compose(transform_list)


def combine_images(image1, image2):
    image1 = Image.fromarray(image1)
    image1 = image1.resize((128, 256))
    image1 = np.array(image1)
    
    image2 = Image.fromarray(image2)
    image2 = image2.resize((128, 256))
    image2 = np.array(image2)

    image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
    return np.concatenate((image1, image2), axis=1)
