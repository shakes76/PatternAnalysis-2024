import pathlib
import logging
from typing import override

import torch
from torchvision import io, transforms
import pandas as pd
from torch.utils.data import Dataset

IMAGE_NAME = "image_name"

logger = logging.getLogger(__name__)


class TumorPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, image_path: pathlib.Path, pair_df: pd.DataFrame) -> None:
        self._image_path = image_path
        self._pair_df = pair_df

    def __len__(self) -> int:
        return len(self._pair_df)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image1_name, image2_name, target = self._pair_df.iloc[index]
        image1 = io.read_image(self._image_path / f"{image1_name}.jpg") / 255
        image2 = io.read_image(self._image_path / f"{image2_name}.jpg") / 255
        image1 = _transform(image1)
        image2 = _transform(image2)

        return image1, image2, target


class TumorClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, image_path: pathlib.Path, meta_df: pd.DataFrame) -> None:
        self._image_path = image_path
        self._meta_df = meta_df

    def __len__(self) -> int:
        return len(self._meta_df)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        observation_data = self._meta_df.iloc[index]
        image_name = observation_data[IMAGE_NAME]
        target = observation_data[IMAGE_NAME]

        image = io.read_image(self._image_path / f"{image_name}.jpg") / 255
        image = _transform(image)

        return image, target


_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
