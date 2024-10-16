import os
import glob
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandFlipd,
    Resized, RandRotated, Rand3DElasticd, RandSpatialCropd, EnsureChannelFirstd,

)
# example_filename = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon\Case_004_Week0_SEMANTIC_LFOV.nii.gz"
# example_label = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon\Case_004_Week0_LFOV.nii.gz"
#
# img = nib.load(example_filename)
# img_data = img.get_fdata()
# img_affine = img.affine
# label = nib.load(example_label)
# label_data = label.get_fdata()
# print(img)

class MRIDataset(Dataset):
    def __init__(self, image_paths, label_paths=None, transform=None, norm_image=False, categorical=False,
                 dtype=np.float32, orient=False, early_stop=False):
        self.image_paths = image_paths
        self.label_paths = label_paths  # 如果有标签
        self.transform = transform
        self.norm_image = norm_image
        self.categorical = categorical
        self.dtype = dtype
        self.orient = orient
        self.early_stop = early_stop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像数据
        image = nib.load(self.image_paths[idx])
        image_data = image.get_fdata().astype(self.dtype)  # 形状：[H, W, D]

        # 如果需要，调整维度顺序为 [C, H, W, D]
        image_data = np.expand_dims(image_data, axis=0)  # 增加通道维度，形状：[1, H, W, D]

        # 归一化
        if self.norm_image:
            image_data = (image_data - image_data.mean()) / image_data.std()

        # 加载标签数据
        if self.label_paths is not None:
            label = nib.load(self.label_paths[idx])
            label_data = label.get_fdata().astype(np.uint8)  # 形状：[H, W, D]
            label_data = np.expand_dims(label_data, axis=0)  # 增加通道维度，形状：[1, H, W, D]
        else:
            label_data = None

        # 创建样本
        sample = {'image': torch.tensor(image_data)}
        if label_data is not None:
            sample['label'] = torch.tensor(label_data)

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample

    def to_channels(self, arr):
        channels = np.unique(arr)
        res = np.zeros(arr.shape + (len(channels),), dtype=np.uint8)
        for idx, c in enumerate(channels):
            res[..., idx][arr == c] = 1
        return res


# 定义数据增强的变换
train_transforms = Compose(
    [
        Rand3DElasticd(keys=("image", "label"),sigma_range=(5, 8),magnitude_range=(100, 200),prob=0.5,
                       mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=("image", "label"),roi_size=(224, 224, 96),random_size=False),
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=(0, 1, 2)),
        RandRotated(keys=("image", "label"), range_x=(-10, 10), range_y=(-10, 10), prob=0.5, mode='nearest'),
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys=("image",), dtype=torch.float32),
        EnsureTyped(keys=("label",), dtype=torch.long),
    ]
)

val_transforms = Compose(
    [
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys=("image",), dtype=torch.float32),
        EnsureTyped(keys=("label",), dtype=torch.long),
    ]
)

if __name__ == '__main__':
    # 指定图像和标签的文件夹路径
    train_image_folder = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon"
    train_label_folder = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon"
    # train_image_folder = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    # train_label_folder = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
    # 获取所有图像和标签路径
    train_image_paths = sorted(glob.glob(os.path.join(train_image_folder, "*.nii*")))
    train_label_paths = sorted(glob.glob(os.path.join(train_label_folder, "*.nii*")))

    # 打印一些路径来验证（可选）
    print("Image paths:", train_image_paths[:3])  # 打印前三个路径，检查路径是否正确
    print("Label paths:", train_label_paths[:3])  # 打印前三个路径，检查路径是否正确

    # 构建数据集
    train_dataset = MRIDataset(
        image_paths=train_image_paths,
        label_paths=train_label_paths,
        transform=train_transforms,
        norm_image=True,
        dtype=np.float32
    )
    print(len(train_dataset))

    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 验证 DataLoader 和 Transform 是否正确工作
    for batch in train_loader:
        images, labels = batch['image'], batch['label']
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break