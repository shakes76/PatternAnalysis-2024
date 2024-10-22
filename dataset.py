import torch
import torch.nn.parallel
import torch.utils.data
import utils as utils
import glob
import random

# Initialise the device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def load_data(testing=False, batch_size=16):
    """
        Loads the data from the Nifti files and returns the training, sting and validation data loaders. The
        validation set is taken to be half od the provided training set, as it is important to be validating the model
        in each epoch to judge performance, yet also test the performance of the final model on unseen data.

        Input:
            testing: when True loads a smaller subset of the data for faster testing
        Output:
            train_dataloader: the DataLoader for the training data
            test_dataloader: the DataLoader for the testing data
            val_dataloader: the DataLoader for the validation data
    """

    train_folder_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
    test_folder_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

    train_file_list = sorted(glob.glob(f"{train_folder_path}/**.nii.gz", recursive=True))
    test_file_list = sorted(glob.glob(f"{test_folder_path}/**.nii.gz", recursive=True))

    if testing:
        train_file_list = train_file_list[:500]
        test_file_list = test_file_list[:200]

    val_file_list = random.sample(test_file_list, 50)
    test_file_list = [f for f in test_file_list if f not in val_file_list]
    
    train_dataset = utils.load_data_2d(train_file_list, norm_image=True) # Normalise data to train more efficiently
    test_dataset = utils.load_data_2d(test_file_list, norm_image=True)
    val_dataset = utils.load_data_2d(val_file_list, norm_image=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
    return train_dataloader, test_dataloader, val_dataloader

def load_test_data(batch_size=16):
    test_folder_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

    test_file_list = sorted(glob.glob(f"{test_folder_path}/**.nii.gz", recursive=True))
    
    test_dataset = utils.load_data_2d(test_file_list)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
   
    return test_dataloader