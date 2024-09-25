import pathlib

import pandas as pd
from torch.utils.data import Dataset

class TumorDataset(Dataset):
    def __init__(self, image_path: pathlib.Path, meta_path: pathlib.Path) -> None:
        self._data_path = data_path
        self._meta_data = pd.read_csv(meta_path)
        
        
