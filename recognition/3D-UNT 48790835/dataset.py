import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class NiftiDataset(Dataset):
    """Nifti格式数据集"""
    def __init__(self, image_files, label_files=None, transform=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform

        # 确保图像和标签数量一致
        if self.label_files is not None:
            assert len(self.image_files) == len(self.label_files), "图像和标签文件数量不一致！"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        image = nib.load(self.image_files[idx]).get_fdata().astype(np.float32)
        # 处理异常值，避免NaN
        image = np.nan_to_num(image)
        # 标准化
        if np.std(image) != 0:
            image = (image - np.mean(image)) / np.std(image)
        else:
            image = image - np.mean(image)
        # 转换为张量
        image = torch.from_numpy(image).unsqueeze(0)

        if self.label_files is not None:
            # 加载标签
            label = nib.load(self.label_files[idx]).get_fdata().astype(np.uint8)
            label = np.nan_to_num(label)
            # 转换为张量
            label = torch.from_numpy(label).long()
            return image, label
        else:
            return image