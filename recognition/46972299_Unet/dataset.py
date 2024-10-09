"""
File for loading the required dataset for the Unet
"""
import torch
from torch.utils.data import Dataset
import torchio as tio
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
TOP_LEVEL_DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/"
SEMANTIC_LABELS = "semantic_labels_only/"
SEMANTIC_MRS = "semantic_MRs/"

class ProstateDataset(Dataset):
    FILE_TYPE = "*.nii.gz"

    def __init__(self, image_dir: str, mask_dir: str, num_classes: int, num_load: int = None) -> None:
        images = [image_dir + file.name for file in Path(image_dir).glob(self.FILE_TYPE) if file.is_file()]
        masks = [mask_dir + file.name for file in Path(mask_dir).glob(self.FILE_TYPE) if file.is_file()]

        if num_load is None:
            num_load = len(images)

        print(f"Loading {num_load} images from {image_dir}")
        self.image_3D_data = load_data_3D(images[:num_load]) 
        print(f"Loading {num_load} masks from {mask_dir}")
        self.mask_3D_data = load_data_3D(masks[:num_load])

        self.transform = tio.Compose([
            tio.RescaleIntensity((0, 1)),
            tio.ZNormalization(),
            tio.OneHot(num_classes)
        ])

        self.processed = []

    def __len__(self) -> int:
        return len(self.image_3D_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image_3D_data[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        mask = self.mask_3D_data[idx]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask)
        )
        subject = self.transform(subject)
        self.processed.append(subject)

        return subject['image'].data, subject['mask'].data