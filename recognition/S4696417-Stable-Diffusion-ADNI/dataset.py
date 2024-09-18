import cv2, os
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
        img = cv2.imread(os.path.join(self.path, self.files[index]))
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img
    
def get_dataloader(path, batch_size, transform=None):
    """
    Return a dataloader with the given path and batch size

    Args:
        path (str): Path to the directory containing the images
        batch_size (int): Batch size
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    dataset = ADNIDataset(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader