import numpy as np
from utils import load_image_and_label_3D, get_images, collate_batch, load_data_3D
from monai.transforms import (Compose, ToTensord, Spacingd, ScaleIntensityRanged, CropForegroundd,
                              Orientationd, RandCropByPosNegLabeld)
from monai.data import list_data_collate
from torch.utils.data import Dataset, DataLoader
from config import NUM_WORKERS, EARLY_STOP

# test other transforms
train_transforms = Compose(
    [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        ToTensord(keys=["image", "label"], device="cpu", track_meta=False),
    ]
)
val_transforms = Compose(
    [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        ToTensord(keys=["image", "label"], device="cpu", track_meta=False)
    ]
)

transforms_dict = {
    'train': train_transforms,
    'valid': val_transforms
}


class MRIDataset(Dataset):
    """
    Custom Dataset class for loading MRI images and masks with MONAI transformations.
    """

    def __init__(self, image_files, label_files, mode: str):
        """
        Initialize the dataset by loading file paths and transformations.

        :param mode: Dataset split type ('train', 'valid')
        """
        self.transform = transforms_dict.get(mode)
        self.images = load_data_3D(image_files, early_stop=EARLY_STOP)
        self.labels = load_data_3D(label_files, early_stop=EARLY_STOP)

    def __len__(self):
        # Return the number of images in the dataset.
        return self.images.shape[0]

    def __getitem__(self, index):
        """
        Load an image and its corresponding mask, apply transformations.

        :param index: Index of the item to retrieve
        :return: Dictionary with transformed image and mask
        """
        # Get image and label at index
        img = self.images[index]  # Retrieve the image at the specified index
        label = self.labels[index]  # Retrieve the label at the specified index

        # Move channel dimension to the front of the label (channels, rows, cols, depth)
        label = np.transpose(label, (3, 0, 1, 2))

        # Add a new channel dimension to the image (1, rows, cols, depth)
        img = np.expand_dims(img, axis=0)

        # Apply transformations
        data = self.transform({'image': img, 'label': label})

        return data


def get_dataloaders(train_batch, val_batch) -> tuple[DataLoader, DataLoader, DataLoader]:
    image_files, mask_files = get_images()

    num_samples = len(image_files)
    np.random.seed(42)
    indices = np.random.permutation(num_samples)

    # Define split sizes (80% train, 10% val, 10% test)
    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

    # use numpy advanced indexing (pass a list of indices)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]

    train_images, train_masks = image_files[train_idx], mask_files[train_idx]
    val_images, val_masks = image_files[val_idx], mask_files[val_idx]
    test_images, test_masks = image_files[test_idx], mask_files[test_idx]

    # get datasets
    train_ds = MRIDataset(train_images, train_masks, mode='train')
    val_ds = MRIDataset(val_images, val_masks, mode='valid')
    test_ds = MRIDataset(test_images, test_masks, mode='valid')

    # TODO: reproducibility, may need to add worker_init_fn to dataloaders
    # get dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=train_batch, num_workers=NUM_WORKERS, collate_fn=collate_batch,
                                  shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=val_batch, num_workers=NUM_WORKERS, collate_fn=collate_batch,
                                shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_ds, batch_size=val_batch, num_workers=NUM_WORKERS, collate_fn=collate_batch,
                                 shuffle=True, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_test_dataloader(batch_size):
    image_files, mask_files = get_images()
    num_samples = len(image_files)
    np.random.seed(42)
    indices = np.random.permutation(num_samples)

    split = int(0.9 * num_samples)
    test_idx = indices[split:]
    test_images, test_masks = image_files[test_idx], mask_files[test_idx]
    test_ds = MRIDataset(test_images, test_masks, mode='valid')
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True,
                                 collate_fn=list_data_collate)
    return test_dataloader
