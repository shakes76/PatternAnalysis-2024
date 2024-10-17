"""
File for loading the required dataset for the Unet

@author Carl Flottmann
"""
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from pathlib import Path
from utils import load_data_3D

# HipMRO_Study_open
# --> semantic_labels_only
# --> semantic_MRs
# --> keras_slices_data
# ------> keras_slices_seg_test
# ------> keras_slices_train
# ------> keras_slices_seg_validate
# ------> keras_slices_seg_train
# ------> keras_slices_validate
RANGPUR_DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/"
SEMANTIC_LABELS = "semantic_labels_only"
SEMANTIC_MRS = "semantic_MRs"
LOCAL_DATA_DIR = ".\\data\\"
LINUX_SEP = "/"
WINDOWS_SEP = "\\"


class ProstateDataset(Dataset):
    FILE_TYPE = "*.nii.gz"

    def __init__(self, image_dir: str, mask_dir: str, num_classes: int, num_load: int = None) -> None:
        images = [
            image_dir + file.name for file in Path(image_dir).glob(self.FILE_TYPE) if file.is_file()]
        masks = [
            mask_dir + file.name for file in Path(mask_dir).glob(self.FILE_TYPE) if file.is_file()]

        if num_load is None:
            num_load = len(images)

        print(f"Loading {num_load} images from {image_dir}")
        self.image_3D_data = load_data_3D(images[:num_load])
        print(f"Loading {num_load} masks from {mask_dir}")
        self.mask_3D_data = load_data_3D(masks[:num_load])

    def __len__(self) -> int:
        return len(self.image_3D_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image_3D_data[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # Rescale intensity to (0, 1)
        image = (image - image.min()) / (image.max() - image.min())
        # Normalisation
        image = (image - image.mean()) / image.std()

        mask = self.mask_3D_data[idx]
        mask = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)

        return image, mask
