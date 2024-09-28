import pathlib

import pandas as pd
from torch.utils.data import Dataset

class TumorDataset(Dataset):
    def __init__(self, image_path: pathlib.Path, meta_df: pd.DataFrame) -> None:
        self._image_path = image_path
        self._metda_df = meta_df
        
        
