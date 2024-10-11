import pathlib
import logging
from typing import override

import torch
from torchvision import io, transforms
from torchvision.transforms import v2
import pandas as pd
from torch.utils.data import Dataset

IMAGE_NAME = "image_name"
TARGET = "target"

logger = logging.getLogger(__name__)


class TumorClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, image_path: pathlib.Path, meta_df: pd.DataFrame, transform: bool = True
    ) -> None:
        self._image_path = image_path
        self._meta_df = meta_df
        self._transform = transform

    def __len__(self) -> int:
        return len(self._meta_df)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        observation_data = self._meta_df.iloc[index]
        image_name = observation_data[IMAGE_NAME]
        target = observation_data[TARGET]

        image = io.read_image(self._image_path / f"{image_name}.jpg") / 255

        if self._transform:
            image = _augmentations(image)

        return image, target


class AllTumorDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, image_path: pathlib.Path) -> None:
        self._image_paths = sorted(list(image_path.iterdir()))
        self._len = len(self._image_paths)

    def __len__(self) -> int:
        return self._len

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return (
            transforms.CenterCrop(224)(
                transforms.Resize(256)(io.read_image(self._image_paths[index]) / 255)
            ),
            self._image_paths[index].stem,
        )


_augmentations = transforms.Compose(
    [
        v2.RandomCrop(size=(224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ]
)
