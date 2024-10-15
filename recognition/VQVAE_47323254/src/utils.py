from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import Compose
from torchvision import transforms
import yaml


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
