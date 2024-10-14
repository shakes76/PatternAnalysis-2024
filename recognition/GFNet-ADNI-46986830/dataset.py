import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Any
from sklearn.model_selection import train_test_split

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
        print("Loading ADNI dataset...")
        print("Directories:")
        print(ad_dir)
        print(nc_dir)

    # Collect file paths and labels
    ad_image_paths = [os.path.join(ad_dir, fname) for fname in os.listdir(ad_dir) if fname.endswith('.jpeg')]
    nc_image_paths = [os.path.join(nc_dir, fname) for fname in os.listdir(nc_dir) if fname.endswith('.jpeg')]

    ad_labels = [1] * len(ad_image_paths)  # AD label is 1
    nc_labels = [0] * len(nc_image_paths)  # NC label is 0

    image_paths = ad_image_paths + nc_image_paths
    labels = ad_labels + nc_labels

    # Split data by subjects (assuming filename contains the subject ID like 1003730)
    # Extract subject IDs from filenames
    subject_ids = [os.path.basename(p).split('_')[0] for p in image_paths]

    if verbose:
        print("Number of subjects:", len(subject_ids))

    # Split subject-wise
    train_subjects, val_subjects = train_test_split(list(set(subject_ids)), test_size=val_size, random_state=42)

    train_paths = [p for p in image_paths if os.path.basename(p).split('_')[0] in train_subjects]
    val_paths = [p for p in image_paths if os.path.basename(p).split('_')[0] in val_subjects]

    train_labels = [labels[i] for i, p in enumerate(image_paths) if os.path.basename(p).split('_')[0] in train_subjects]
    val_labels = [labels[i] for i, p in enumerate(image_paths) if os.path.basename(p).split('_')[0] in val_subjects]

    


if __name__ == "__main__":

    adni_dir = "/home/reuben/Documents/GFNet_testing/ADNI_AD_NC_2D/AD_NC"
    adni_data_load(adni_dir, verbose=True)
