import numpy as np
from utils import get_images, collate_batch, load_image_and_label_3D
from monai.transforms import (Compose, ToTensord, RandCropByLabelClassesd, RandFlipd, NormalizeIntensityd, Resized)
from torch.utils.data import Dataset, DataLoader

train_transforms = Compose(
    [
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True),
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label", image_key="image",
                                spatial_size=(96, 96, 48), num_samples=6, ratios=[1, 2, 3, 4, 4, 4]),
        ToTensord(keys=["image", "label"], device="cpu", track_meta=False),
    ]
)
val_transforms = Compose(
    [
        NormalizeIntensityd(keys='image', nonzero=True),
        # most images already in this shape (256, 256, 128), resized is for consistency
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        ToTensord(keys=["image", "label"], device="cpu", track_meta=False),
    ]
)

transforms_dict = {
    'train': train_transforms,
    'valid': val_transforms
}


class MRIDataset(Dataset):
    """
    Custom Dataset class for loading MRI images_files and label_files with MONAI transformations.
    """

    def __init__(self, images_files, label_files, mode: str):
        self.transform = transforms_dict.get(mode)
        self.image_files = images_files
        self.label_files = label_files

    def __len__(self):
        # Return the number of images_files in the dataset.
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Load an image and its corresponding mask, apply transformations.

        :param index: Index of the item to retrieve
        :return: Dictionary with transformed image and mask
        """
        img_and_mask = load_image_and_label_3D(self.image_files[index], self.label_files[index])
        # Load image and segmentation
        data = {'image': img_and_mask[0], 'label': img_and_mask[1]}
        data = self.transform(data)  # Apply transformations
        return data

def get_dataloaders(batch_size=1) -> tuple[DataLoader, DataLoader]:
    image_files, label_files = get_images()

    num_samples = len(image_files)
    np.random.seed(42)
    indices = np.random.permutation(num_samples)

    # Define split sizes (80% train, 10% val, 10% test)
    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

    # use numpy advanced indexing (pass a list of indices)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]

    train_image_files, train_label_files = image_files[train_idx], label_files[train_idx]
    val_image_files, val_label_files = image_files[val_idx], label_files[val_idx]

    # get datasets
    train_ds = MRIDataset(train_image_files, train_label_files, mode='train')
    val_ds = MRIDataset(val_image_files, val_label_files, mode='valid')

    # get dataloaders
    # train_dataloader = DataLoader(train_ds, batch_size=train_batch, num_workers=NUM_WORKERS, collate_fn=collate_batch,
    #                               shuffle=True)
    # val_dataloader = DataLoader(val_ds, batch_size=val_batch, num_workers=NUM_WORKERS, collate_fn=collate_batch,
    #                             shuffle=True)
    train_dataloader = DataLoader(train_ds, batch_size=1, num_workers=0, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_ds, batch_size=1, num_workers=0)

    return train_dataloader, val_dataloader


def get_test_dataloader():
    image_files, mask_files = get_images()
    num_samples = len(image_files)
    np.random.seed(42)
    indices = np.random.permutation(num_samples)

    split = int(0.9 * num_samples)
    test_idx = indices[split:]
    test_images, test_masks = image_files[test_idx], mask_files[test_idx]
    test_ds = MRIDataset(test_images, test_masks, mode='valid')
    test_dataloader = DataLoader(test_ds, batch_size=1, num_workers=0)
    return test_dataloader
