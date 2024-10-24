"""
Utility functions useful across the multiple files.

Author: Kevin Gu
"""
import torch
import platform

def get_device() -> str:
    """
    Utility function to get the device to run on.

    Parameters: None
    Returns: The device on which code will be run
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("Could not find Cuda GPU, using CPU")
    return device

def get_dataset_root() -> str:
    """
    Get the root of ADNI dataset by checking what OS is being run.

    Parameters: None
    Returns: Root directory of ADNI dataset
    """
    if platform.system() == "Windows":
        root_dir = 'ADNI_AD_NC_2D/AD_NC'
    else:
        root_dir = '/home/groups/comp3710/ADNI/AD_NC'
    return root_dir