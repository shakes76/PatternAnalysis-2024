import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ADNI_DataLoader

def main():
    """
    Constants
    """
    IMAGESIZE = 240
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/"
    EPOCHS = 100
    lr = 0.001

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed to make replicable
    seed = 42
    torch.manual_seed(seed)

    # Get the data
    print("Loading ADNI brain data")
    full_dataHandler = ADNI_DataLoader(rootData=ROOTDATAPATH, imageSize=IMAGESIZE)
    train_dataloader = full_dataHandler.get_dataloader(data_type="train")
    test_dataloader = full_dataHandler.get_dataloader(data_type="test")

if __name__ == "__main__":
    main()