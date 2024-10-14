import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Any
# from sklearn.model_selection import train_test_split

class ADNIDataset(Dataset):
    def __init__(self) -> None:
        # load data
        pass
    
    def __getitem__(self, index) -> Any:
        # get data
        pass

    def __len__(self):
        # get number of samples
        pass

def adni_data_load(root_dir, val_size=0.2, batch_size=32, verbose = False):
    """ 
    Args:
        root_dir (str): the root directory containing the ADNI data
        val_size (float): amount of the training set to use for validation
        batch_size (int)
        
    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
    """

    # Directories for AD and NC (Normal Controls)
    ad_dir = os.path.join(root_dir, 'train', 'AD')
    nc_dir = os.path.join(root_dir, 'train', 'NC')

    if verbose:
        print("ADNI data directories:")
        print(ad_dir)
        print(nc_dir)

    # Collect file paths and labels
    ad_image_paths = [os.path.join(ad_dir, fname) for fname in os.listdir(ad_dir) if fname.endswith('.jpeg')]
    nc_image_paths = [os.path.join(nc_dir, fname) for fname in os.listdir(nc_dir) if fname.endswith('.jpeg')]

    ad_labels = [1] * len(ad_image_paths)  # AD label is 1
    nc_labels = [0] * len(nc_image_paths)  # NC label is 0

    image_paths = ad_image_paths + nc_image_paths
    labels = ad_labels + nc_labels

    print(image_paths)
    print(labels)
    


if __name__ == "__main__":

    adni_dir = "/home/reuben/Documents/GFNet_testing/ADNI_AD_NC_2D/AD_NC"
    adni_data_load(adni_dir, verbose=True)
