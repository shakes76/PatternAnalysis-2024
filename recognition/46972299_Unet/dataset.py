"""
File for loading the required dataset for the Unet

@author Carl Flottmann
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from pathlib import Path
from utils import load_data_3D, cur_time
import re
import math

# HipMRI_Study_open
# --> semantic_labels_only
# --> semantic_MRs
RANGPUR_DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/"
SEMANTIC_LABELS = "semantic_labels_only"
SEMANTIC_MRS = "semantic_MRs"
LOCAL_DATA_DIR = ".\\data\\"
LINUX_SEP = "/"
WINDOWS_SEP = "\\"


class ProstateDataset(Dataset):
    def __init__(self, images: list[str], masks: list[str], start: int, end: int, start_t: float = None) -> None:

        print(f"[{cur_time(start_t) if start_t is not None else "i"}] Loading {
              end - start} images")
        self.image_3D_data = load_data_3D(images[start:end])
        print(f"[{cur_time(start_t) if start_t is not None else "i"}] Loading {
              end - start} masks")
        self.mask_3D_data = load_data_3D(masks[start:end])

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


class ProstateLoader(DataLoader):
    FILE_TYPE = "*.nii.gz"
    # file names look like <letter><three-digit-number>_Week<number>_SEMANTIc or LFOV
    PREFIX_MATCH = r'([A-Z]+\d+_Week\d+)'
    TRAIN_PORTION = 0.65  # 65% of images we'll train on
    VALIDATE_PORTION = 0.20  # 20% of images we'll validate on
    # rest (15%) will be reserved for testing
    MIN_LOAD = 7  # must load at least 7 images total for this split to work

    def __init__(self, image_dir: str, mask_dir: str, num_load: int = None, start_t: float = None, **kwargs) -> None:
        if num_load < self.MIN_LOAD:
            raise ValueError(f"Must load at least {
                             self.MIN_LOAD} images and masks")

        self.start_t = start_t
        self.kwargs = kwargs

        image_names = [file.name for file in Path(
            image_dir).glob(self.FILE_TYPE) if file.is_file()]
        mask_names = [file.name for file in Path(
            mask_dir).glob(self.FILE_TYPE) if file.is_file()]

        # ensure each index of the dataset matches in the images and masks
        image_names = sorted(image_names, key=self.__match_dataset_names)
        mask_names = sorted(mask_names, key=self.__match_dataset_names)

        # add on the directory so we can load properly
        self.images = [image_dir + name for name in image_names]
        self.masks = [mask_dir + name for name in mask_names]

        if num_load is None or num_load > len(self.images):
            num_load = len(self.images)

        self.num_train_images = math.ceil(self.TRAIN_PORTION * num_load)
        self.num_validate_images = math.floor(self.VALIDATE_PORTION * num_load)
        self.num_load = num_load
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def get_num_load(self) -> int:
        return self.num_load

    def train(self) -> DataLoader:
        dataset = ProstateDataset(
            self.images, self.masks, 0, self.num_train_images, self.start_t)
        return DataLoader(dataset, **self.kwargs)

    def validate(self) -> DataLoader:
        dataset = ProstateDataset(
            self.images, self.masks, self.num_train_images, self.num_train_images + self.num_validate_images, self.start_t)
        return DataLoader(dataset, **self.kwargs)

    def test(self) -> DataLoader:
        dataset = ProstateDataset(
            self.images, self.masks, self.num_train_images + self.num_validate_images, self.num_load, self.start_t)
        return DataLoader(dataset, **self.kwargs)

    def __match_dataset_names(self, filename: str) -> str:
        match = re.search(self.PREFIX_MATCH, filename)
        if match:
            return match.group(1)  # the match on the prefix
        else:
            return None

    def state_dict(self) -> dict:
        return {
            'image_dir': self.image_dir,
            'mask_dir': self.mask_dir,
            'num_load': self.num_load,
            'kwargs': self.kwargs
        }

    @staticmethod
    def load_state_dict(state_dict: dict, start_t: int = None) -> ProstateLoader:
        return ProstateLoader(state_dict['image_dir'], state_dict['mask_dir'], state_dict['num_load'], start_t, **state_dict['kwargs'])
