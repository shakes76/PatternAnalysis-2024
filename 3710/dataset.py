import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import tensorflow_addons as tfa
import torchvision.transforms as transforms


mrs_dir = "/Users/zhangxiangxu/Downloads/3710_data/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
labels_dir = "/Users/zhangxiangxu/Downloads/3710_data/data/HipMRI_study_complete_release_v1/semantic_labels_anon"

# 自定义数据集类，用于加载 MRI 图像和对应的标签
class MRIDataset(Dataset):
    def __init__(self, mrs_dir, labels_dir, transform=None):
        self.mrs_dir = mrs_dir  # MRI 图像文件夹路径
        self.labels_dir = labels_dir  # 标签文件夹路径
        self.mr_files = sorted(os.listdir(mrs_dir))  # 获取 MRI 文件列表并排序
        self.label_files = sorted(os.listdir(labels_dir))  # 获取标签文件列表并排序
        self.transform = transform  # 数据增强变换


    def __getitem__(self, idx):
        # 加载 MRI 图像
        mr_path = os.path.join(self.mrs_dir, self.mr_files[idx])
        mr_image = nib.load(mr_path).get_fdata()  # 使用 nibabel 加载 NIfTI 文件并获取数据
        # 对 MRI 图像进行归一化处理，将像素值缩放到 [0, 1] 范围内
        mr_image = (mr_image - np.min(mr_image)) / (np.max(mr_image) - np.min(mr_image))
        mr_image = np.expand_dims(mr_image, axis=0)  # 添加通道维度 (C, D, H, W)

        # 加载标签图像
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label_image = nib.load(label_path).get_fdata()  # 使用 nibabel 加载标签文件并获取数据
        label_image = np.expand_dims(label_image, axis=0)  # 添加通道维度 (C, D, H, W)

        # 将数据转换为 PyTorch 张量
        mr_image = torch.tensor(mr_image, dtype=torch.float32)  # MRI 图像转换为 float32 类型
        label_image = torch.tensor(label_image, dtype=torch.long)  # 标签图像转换为 long 类型

        # 如果有数据增强变换，则对 MRI 和标签进行变换
        if self.transform:
            mr_image = self.transform(mr_image)
            label_image = self.transform(label_image)

        return mr_image, label_image  # 返回 MRI 图像和对应的标签
    
