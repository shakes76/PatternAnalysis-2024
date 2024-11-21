"""
File containing custom data loaders for the HipMRI Prostate 3D MRI Dataset, and relevant constants for the Rangpur and windows local environment.

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
    """
    Class that will load data and apply appropriate transforms for the HipMRI Prostate 3D MRI Dataset. Training transforms are available
    as a constant.

    Inherits:
        Dataset: pytorch dataset class.
    """
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ElasticTransform(),
        transforms.RandomAffine(0)
    ])

    def __init__(self, images: list[str], masks: list[str], start: int, end: int, start_t: float = None, train: bool = False) -> None:
        """
        Initialise and load the images for this dataset.

        Args:
            images (list[str]): a list of image names in the same directory to load.
            masks (list[str]): a list of corresponding mask names in the same directory to load. Indexes must correspond to the image indexes.
            start (int): the index in the images and masks to start loading from.
            end (int): the index in the images and masks to stop loading from.
            start_t (float, optional): will print out the time since start_t if supplied on print statements. Defaults to None.
            train (bool, optional): indicate if this dataset is to be used for training, for transformation purposes. Defaults to False.
        """

        print(f"[{cur_time(start_t) if start_t is not None else "i"}] Loading {
              end - start} images")
        self.image_3D_data = load_data_3D(images[start:end])
        print(f"[{cur_time(start_t) if start_t is not None else "i"}] Loading {
              end - start} masks")
        self.mask_3D_data = load_data_3D(masks[start:end])
        self.train = train

    def __len__(self) -> int:
        """
        Return the number of image-mask pairs this dataset will load.

        Returns:
            int: number of image-mask pairs in this dataset.
        """
        return len(self.image_3D_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations and retrieve an image-mask pair from the dataset.

        Args:
            idx (int): the index of the image-mask pair to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: an image, mask 4D tensor.
        """
        image = self.image_3D_data[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # Rescale intensity to (0, 1)
        image = (image - image.min()) / (image.max() - image.min())

        mask = self.mask_3D_data[idx]
        mask = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)

        if self.train:
            image = self.TRAIN_TRANSFORM(image)
            mask = self.TRAIN_TRANSFORM(mask)

        # Normalisation
        image = (image - image.mean()) / image.std()

        return image, mask


class ProstateLoader(DataLoader):
    """
    Custom data loader for the the HipMRI Prostate 3D MRI Dataset. Will perform the train, validate, and test split, and can
    be saved with the model.

    Inherits:
        DataLoader: pytorch DataLoader class.
    """
    FILE_TYPE = "*.nii.gz"
    # file names look like <letter><three-digit-number>_Week<number>_SEMANTIc or LFOV
    PREFIX_MATCH = r'([A-Z]+\d+_Week\d+)'
    TRAIN_PORTION = 0.65  # 65% of images we'll train on
    VALIDATE_PORTION = 0.20  # 20% of images we'll validate on
    # rest (15%) will be reserved for testing
    MIN_LOAD = 7  # must load at least 7 images total for this split to work

    def __init__(self, image_dir: str, mask_dir: str, num_load: int = None, start_t: float = None, **kwargs) -> None:
        """
        Initialise the data loader.

        Args:
            image_dir (str): directory to load images from.
            mask_dir (str): directory to load corresponding masks from.
            num_load (int, optional): number of image-mask pairs to load if not all. Defaults to None, which will load all.
            start_t (float, optional): will print out the time since start_t if supplied on print statements. Defaults to None.
            kwargs: any extra arguments that would usually be given to the pytorch DataLoader, e.g. batch size, number of workers, etc.

        Raises:
            ValueError: if the number of images to load is too small and train, validate, and test splits cannot be performed.
        """
        if num_load is not None:
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
        """
        Get the number of image-mask pairs this data loader has been set to load.

        Returns:
            int: number of image-mask pairs this data loader has been set to load.
        """
        return self.num_load

    def train(self) -> DataLoader:
        """
        Retrieve a data loader for the train set.

        Returns:
            DataLoader: data loader for the train set.
        """
        dataset = ProstateDataset(
            self.images, self.masks, 0, self.num_train_images, self.start_t, True)
        return DataLoader(dataset, **self.kwargs)

    def validate(self) -> DataLoader:
        """
        Retrieve a data loader for the validate set.

        Returns:
            DataLoader: data loader for the validate set.
        """
        dataset = ProstateDataset(
            self.images, self.masks, self.num_train_images, self.num_train_images + self.num_validate_images, self.start_t)
        return DataLoader(dataset, **self.kwargs)

    def validate_size(self) -> int:
        """
        Get the number of image-mask pairs this data loader has been set to load for validation.

        Returns:
            int: number of image-mask pairs this data loader has been set to load for validation.
        """
        return self.num_validate_images

    def test(self) -> DataLoader:
        """
        Retrieve a data loader for the test set.

        Returns:
            DataLoader: data loader for the test set.
        """
        dataset = ProstateDataset(
            self.images, self.masks, self.num_train_images + self.num_validate_images, self.num_load, self.start_t)
        return DataLoader(dataset, **self.kwargs)

    def test_size(self) -> int:
        """
        Get the number of image-mask pairs this data loader has been set to load for testing.

        Returns:
            int: number of image-mask pairs this data loader has been set to load for testing.
        """
        return self.num_load - self.num_validate_images - self.num_train_images

    def __match_dataset_names(self, filename: str) -> str:
        """
        Key for sorting the dataset image-mask pairs based on their file name.

        Args:
            filename (str): name of the file.

        Returns:
            str: the match on the file name.
        """
        match = re.search(self.PREFIX_MATCH, filename)
        if match:
            return match.group(1)  # the match on the prefix
        else:
            return None

    def state_dict(self) -> dict:
        """
        Retrieve a pytorch saveable dictionary with the data laoder state.

        Returns:
            dict: pytorch saveable dictionary with the data laoder state.
        """
        return {
            'image_dir': self.image_dir,
            'mask_dir': self.mask_dir,
            'num_load': self.num_load,
            'kwargs': self.kwargs
        }

    @staticmethod
    def load_state_dict(state_dict: dict, start_t: int = None) -> ProstateLoader:
        """
        Load a state dictionary and retrieve the data loader object.

        Args:
            state_dict (dict): pytorch saveable dictionary with the data laoder state.
            start_t (float, optional): will print out the time since start_t if supplied on print statements. Defaults to None.

        Returns:
            ProstateLoader: custom data loader with the state supplied.
        """
        return ProstateLoader(state_dict['image_dir'], state_dict['mask_dir'], state_dict['num_load'], start_t, **state_dict['kwargs'])
