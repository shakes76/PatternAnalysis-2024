import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    将三维数组转换为带有通道的独热编码四维数组。
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for idx, c in enumerate(channels):
        res[..., idx][arr == c] = 1
    return res

def applyOrientation(niftiImage, interpolation='linear', scale=1):
    """
    应用方向和缩放到 NIfTI 图像。

    参数：
    - niftiImage：nibabel NIfTI 图像对象。
    - interpolation：插值方法，'linear' 或 'nearest'。
    - scale：缩放因子。
    """
    # 这里进行重采样或重新取样等操作
    # 由于具体实现取决于您的需求，以下是一个示例（需要根据实际情况修改）：
    data = niftiImage.get_fdata()
    affine = niftiImage.affine

    # 进行缩放（如果需要）
    if scale != 1:
        zoom_factors = [scale, scale, scale]
        data = zoom(data, zoom=zoom_factors, order=1 if interpolation == 'linear' else 0)

    # 创建新的 NIfTI 图像
    new_nifti = nib.Nifti1Image(data, affine)
    return new_nifti

class MedicalDataset3D(Dataset):
    """
    自定义数据集类，用于加载 3D 医学图像。
    """
    def __init__(self, image_paths, label_paths=None, transform=None, normImage=False, categorical=False, dtype=np.float32, orient=False):
        """
        参数：
        - image_paths：图像文件路径列表。
        - label_paths：标签文件路径列表（如果有）。
        - transform：数据增强变换。
        - normImage：是否标准化图像。
        - categorical：是否将标签转换为独热编码。
        - dtype：数据类型。
        - orient：是否应用方向和缩放。
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.orient = orient

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image_nifti = nib.load(self.image_paths[idx])
        if self.orient:
            image_nifti = applyOrientation(image_nifti, interpolation='linear', scale=1)
        image = image_nifti.get_fdata(caching='unchanged')
        if len(image.shape) == 4:
            image = image[:, :, :, 0]
        image = image.astype(self.dtype)
        if self.normImage:
            image = (image - image.mean()) / image.std()
        image = np.expand_dims(image, axis=0)  # 添加通道维度

        # 如果有标签，加载标签
        if self.label_paths:
            label_nifti = nib.load(self.label_paths[idx])
            if self.orient:
                label_nifti = applyOrientation(label_nifti, interpolation='nearest', scale=1)
            label = label_nifti.get_fdata(caching='unchanged')
            if len(label.shape) == 4:
                label = label[:, :, :, 0]
            label = label.astype(self.dtype)
            if self.categorical:
                label = to_channels(label, dtype=self.dtype)
                label = np.moveaxis(label, -1, 0)  # 将通道维度移到前面
            else:
                label = np.expand_dims(label, axis=0)  # 添加通道维度
        else:
            label = None

        # 应用数据增强（如果有）
        if self.transform:
            # 注意，您需要确保 transform 适用于 3D 图像数据
            if label is not None:
                data = {"image": image, "mask": label}
                augmented = self.transform(**data)
                image = augmented["image"]
                label = augmented["mask"]
            else:
                data = {"image": image}
                augmented = self.transform(**data)
                image = augmented["image"]

        # 转换为张量
        image = torch.tensor(image, dtype=torch.float32)
        if label is not None:
            label = torch.tensor(label, dtype=torch.float32)
            return image, label
        else:
            return image

def load_image_paths(data_dir, split='train'):
    """
    获取图像和标签的文件路径列表。

    参数：
    - data_dir：数据集目录。
    - split：数据集划分，'train'、'val' 或 'test'。

    返回：
    - image_paths：图像文件路径列表。
    - label_paths：标签文件路径列表。
    """
    # 根据您的数据组织方式，实现获取文件路径的逻辑
    # dataset.py


def load_image_paths(image_dir, label_dir):
    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    label_paths = sorted([
        os.path.join(label_dir, f) for f in os.listdir(label_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    return image_paths, label_paths

