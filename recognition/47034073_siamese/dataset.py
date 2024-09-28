import pathlib

import torch
from torchvision import io
import pandas as pd
from torch.utils.data import Dataset


class TumorPairDataset(Dataset):
    def __init__(self, image_path: pathlib.Path, pair_df: pd.DataFrame) -> None:
        self._image_path = image_path
        self._pair_df = pair_df

    def __len__(self) -> int:
        return len(self._pair_df)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:
        image1_name, image2_name, target = self._pair_df.iloc[index]
        image1 = io.read_image(self._image_path / f"{image1_name}.jpg") / 255
        image2 = io.read_image(self._image_path / f"{image2_name}.jpg") / 255

        return image1, image2, target
