import cv2, os, torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ADNIDataset(Dataset):
    """
    Dataset for ADNI images

    Args:
        path (str): Path to the directory containing the images
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.path, self.files[index]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            img = self.transform(img)

        timestamp = torch.rand(1)

        return img, timestamp
    
def get_dataloader(path, batch_size, transform=None, val_split=0.2):
    """
    Return a dataloader with the given path and batch size

    Args:
        path (str): Path to the directory containing the images
        batch_size (int): Batch size
        transform (callable, optional): Optional transform to be applied
            on a sample.
        val_split (float, optional): Fraction of the dataset to be used as validation set
    """
    train_size = int((1-val_split)*len(os.listdir(path)))
    val_size = len(os.listdir(path)) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(ADNIDataset(path, transform=transform), [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader