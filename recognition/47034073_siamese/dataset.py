import pathlib
import logging

import torch
from torchvision import io, transforms
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TumorPairDataset(Dataset):
    def __init__(self, image_path: pathlib.Path, pair_df: pd.DataFrame) -> None:
        self._image_path = image_path
        self._pair_df = pair_df

    def __len__(self) -> int:
        return len(self._pair_df)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:
        # logger.debug("%s", str(self._pair_df.iloc[index]))
        image1_name, image2_name, target = self._pair_df.iloc[index]
        # logger.debug("\n%s\n%s\n%s", image1_name, image2_name, str(target))
        image1 = io.read_image(self._image_path / f"{image1_name}.jpg") / 255
        image2 = io.read_image(self._image_path / f"{image2_name}.jpg") / 255
        image1 = _transform(image1)
        image2 = _transform(image2)
        # logger.debug("--%s %s", str(image1.shape), str(image2.shape))

        return image1, image2, target


_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
