import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    Resized,
    EnsureTyped,
)

class MRIDataset(Dataset):
    def __init__(self, image_paths, label_paths=None, transform=None, norm_image=False, categorical=False,
                 orient=False):
        self.image_paths = image_paths
        self.label_paths = label_paths  # 如果有标签
        self.transform = transform
        self.norm_image = norm_image
        self.categorical = categorical
        self.orient = orient

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = nib.load(self.image_paths[idx])
        image_data = image.get_fdata().astype(np.float32)

        # 归一化
        if self.norm_image:
            image_data = (image_data - image_data.mean()) / image_data.std()

        # 如果有标签，加载标签
        if self.label_paths is not None:
            label = nib.load(self.label_paths[idx])
            label_data = label.get_fdata().astype(np.uint8)
            if self.categorical:
                label_data = self.to_channels(label_data)
        else:
            label_data = None

        # 转换为张量并应用任何变换
        sample = {'image': image_data}
        if label_data is not None:
            sample['label'] = label_data

        if self.transform:
            sample = self.transform(sample)

        return sample

    def to_channels(self, arr):
        channels = np.unique(arr)
        res = np.zeros(arr.shape + (len(channels),), dtype=np.uint8)
        for idx, c in enumerate(channels):
            res[..., idx][arr == c] = 1
        return res


# for train split : resize and random flip
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys="image", dtype=torch.float32),
        EnsureTyped(keys="label", dtype=torch.long),
    ]
)

# for test split : just load and normalize
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys="image", dtype=torch.float32),
        EnsureTyped(keys="label", dtype=torch.long),
    ]
)
if __name__=='__main__':
    image_paths = '../../Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon'
    label_paths = '../../Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon'

    # 创建测试数据集和数据加载器
    test_dataset = MRIDataset(image_paths=[image_paths], label_paths=[label_paths], transform=val_transforms,
                              norm_image=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

    # 打印数据集长度，确保数据加载器正常工作
    print(f'测试数据集大小: {len(test_dataset)}')
    for batch_ndx, sample in enumerate(test_dataloader):
        print('test')
        print(sample['image'].shape)
        if 'label' in sample:
            print(sample['label'].shape)
        break

    # 创建训练数据集和数据加载器
    train_dataset = MRIDataset(image_paths=[image_paths], label_paths=[label_paths], transform=train_transforms,
                               norm_image=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    # 打印训练数据集的第一批数据的形状
    for batch_ndx, sample in enumerate(train_dataloader):
        print('train')
        print(sample['image'].shape)
        if 'label' in sample:
            print(sample['label'].shape)
        break