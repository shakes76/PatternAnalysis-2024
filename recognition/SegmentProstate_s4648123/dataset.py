import numpy as np
from utils import get_images, collate_batch, load_image_and_label_3D
from config import RANDOM_SEED
from monai.transforms import (Compose, ToTensord, RandCropByLabelClassesd, RandFlipd, NormalizeIntensityd, Resized)
from torch.utils.data import Dataset, DataLoader

# Define training transformations
train_transforms = Compose(
    [
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True),
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label", image_key="image",
                                spatial_size=(96, 96, 48), num_samples=6),
        ToTensord(keys=["image", "label"], device="cpu", track_meta=False),
    ]
)

# Define validation transformations
val_transforms = Compose(
    [
        NormalizeIntensityd(keys='image', nonzero=True),
        # Resize images for consistency
        Resized(keys=["image", "label"], spatial_size=(256, 256, 128)),
        ToTensord(keys=["image", "label"], device="cpu", track_meta=False),
    ]
)

# Dictionary to store transformations for different modes
transforms_dict = {
    'train': train_transforms,
    'valid': val_transforms
}


class MRIDataset(Dataset):
    """
    A dataset class for loading MRI images and their corresponding labels.

    Args:
        images_files (list): List of file paths to the image files.
        label_files (list): List of file paths to the label files.
        mode (str): Mode indicating 'train' or 'valid', which determines the transformations to apply.
    """

    def __init__(self, images_files, label_files, mode: str):
        self.transform = transforms_dict.get(mode)
        self.image_files = images_files
        self.label_files = label_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image and segmentation mask
        img_and_mask = load_image_and_label_3D(self.image_files[index], self.label_files[index])
        data = {'image': img_and_mask[0], 'label': img_and_mask[1]}
        data = self.transform(data)  # Apply transformations
        return data


def get_dataloaders(batch_size=1) -> tuple[DataLoader, DataLoader]:
    """
    Prepare the training and validation data loaders.
    """
    image_files, label_files = get_images()

    num_samples = len(image_files)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(num_samples)

    # Define split sizes (80% train, 10% val, 10% test)
    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

    # Use numpy advanced indexing to get train and validation indices
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]

    train_image_files, train_label_files = image_files[train_idx], label_files[train_idx]
    val_image_files, val_label_files = image_files[val_idx], label_files[val_idx]

    # Create datasets
    train_ds = MRIDataset(train_image_files, train_label_files, mode='train')
    val_ds = MRIDataset(val_image_files, val_label_files, mode='valid')

    # Create data loaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_batch)

    return train_dataloader, val_dataloader


def get_test_dataloader(batch_size=2) -> DataLoader:
    """
    Prepare the test data loader.
    """
    image_files, mask_files = get_images()
    num_samples = len(image_files)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(num_samples)

    split = int(0.9 * num_samples)
    test_idx = indices[split:]
    test_images, test_masks = image_files[test_idx], mask_files[test_idx]
    test_ds = MRIDataset(test_images, test_masks, mode='valid')
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    return test_dataloader
