import os
import glob
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    EnsureTyped,
    CastToTyped,
    NormalizeIntensityd,
    RandFlipd,
    Lambdad,
    Resized,
    RandGaussianNoised,
    RandGridDistortiond,
    RepeatChanneld,
    Transposed,
    OneOf,
    EnsureChannelFirstd,
    RandLambdad,
    Spacingd,
    FgBgToIndicesd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToDeviced,
    SpatialPadd,

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
        # 直接返回图像和标签的路径
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx] if self.label_paths is not None else None

        # 返回包含路径的字典
        sample = {'image': image_path}
        if label_path is not None:
            sample['label'] = label_path

        # 应用 MONAI 的 transforms
        if self.transform:
            sample = self.transform(sample)

        return sample

    def to_channels(self, arr):
        channels = np.unique(arr)
        res = np.zeros(arr.shape + (len(channels),), dtype=np.uint8)
        for idx, c in enumerate(channels):
            res[..., idx][arr == c] = 1
        return res


# for train split : resize and randomfip
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: (x - x.min()) / (x.max() - x.min())),
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=None),
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys=("image", "label"), dtype=torch.float32),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: (x - x.min()) / (x.max() - x.min())),
        EnsureTyped(keys=("image", "label"), dtype=torch.float32),
    ]
)

if __name__ == '__main__':
    # 指定图像和标签的文件夹路径
    train_image_folder = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon"
    train_label_folder = r"C:\Users\YG\Documents\course\COMP3710\Report\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon"

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
        transform=train_transforms
    )
    print(len(train_dataset))
    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 验证 DataLoader 和 Transform 是否正确工作
    for batch in train_loader:
        images, labels = batch['image'], batch['label']
        print(f"Image batch shape: {images.shape}")  # 应该是 (batch_size, channels, height, width, depth)
        print(f"Label batch shape: {labels.shape}")  # 应该是 (batch_size, channels, height, width, depth)
        break
