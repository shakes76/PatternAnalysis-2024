import os
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    RandFlipd,
    Lambdad,
    Resized,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotate90d,
    RandShiftIntensityd,
)

# Transforms for training data: load, resize, and apply random flips and rotations
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys="image"),  # Normalize intensity
    Lambdad(keys="image", func=lambda x: (x - x.min()) / (x.max() - x.min())),  # Further normalization
    RandRotate90d(keys=("image", "label"), prob=0.5),  # Random 90-degree rotations
    RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=[0]),
    RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=[1]),
    RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=[2]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
    EnsureTyped(keys=("image", "label"), dtype=torch.float32),
])

# Transforms for testing data: only load and normalize
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys="image"),
    Lambdad(keys="image", func=lambda x: (x - x.min()) / (x.max() - x.min())),
    EnsureTyped(keys=("image", "label"), dtype=torch.float32),
])

class MRIDataset_pelvis(Dataset):
    """
    Dataset class for reading pelvic MRI data.
    """

    def __init__(self, mode, dataset_path):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            dataset_path (str): Root directory of the dataset.
        """
        self.mode = mode
        self.train_transform = train_transforms
        self.test_transform = val_transforms

        # Load image and label file paths based on mode
        if self.mode == 'train':
            with open('train_list.txt', 'r') as f:
                select_list = [_.strip() for _ in f.readlines()]
            self.img_list = [os.path.join(dataset_path, 'semantic_MRs_anon', _) for _ in select_list]
            self.label_list = [os.path.join(dataset_path, 'semantic_labels_anon', _.replace('_LFOV', '_SEMANTIC_LFOV'))
                               for _ in select_list]

        elif self.mode == 'test':
            with open('test_list.txt', 'r') as f:
                select_list = [_.strip() for _ in f.readlines()]
            self.img_list = [os.path.join(dataset_path, 'semantic_MRs_anon', _) for _ in select_list]
            self.label_list = [os.path.join(dataset_path, 'semantic_labels_anon', _.replace('_LFOV', '_SEMANTIC_LFOV'))
                               for _ in select_list]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]

        if self.mode == 'train':
            augmented = self.train_transform({'image': img_path, 'label': label_path})
            return augmented['image'], augmented['label']

        if self.mode == 'test':
            augmented = self.test_transform({'image': img_path, 'label': label_path})
            return augmented['image'], augmented['label']


if __name__ == '__main__':
    # Test the dataset
    test_dataset = MRIDataset_pelvis(mode='test', dataset_path=r"path_to_your_dataset")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    print(len(test_dataset))
    for batch_ndx, sample in enumerate(test_dataloader):
        print('test')
        print(sample[0].shape)
        print(sample[1].shape)
        break

    train_dataset = MRIDataset_pelvis(mode='train', dataset_path=r"path_to_your_dataset")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
    for batch_ndx, sample in enumerate(train_dataloader):
        print('train')
        print(sample[0].shape)
        print(sample[1].shape)
        break
