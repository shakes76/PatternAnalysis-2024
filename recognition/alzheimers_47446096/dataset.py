from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

ROOT = "../ADNI/AD_NC/train"
TESTPATH = "/test"
TRAINPATH = "/train"

def getTrainLoader(path: str = ROOT + TRAINPATH, batchSize: int = 128, shuffle: bool = True):
    '''
    Get Pytorch DataLoader with ADNI training DATA

    Input:
        path: str - relative path to training data
        batchSize: int - batch size of the DataLoader
        suffle: bool - DataLoader shuffle option
    Returns:
        Pytorch DataLoader with ADNI training data loaded
    '''
    trainData = ImageFolder(path)
    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle = shuffle)
    return trainLoader

def getTestLoader(path = ROOT + TESTPATH, batchSize = 128, shuffle = True):
    '''
    Get Pytorch DataLoader with ADNI test DATA

    Input:
        path: str - relative path to training data
        batchSize: int - batch size of the DataLoader
        suffle: bool - DataLoader shuffle option
    Returns:
        Pytorch DataLoader with ADNI test data loaded
    '''
    trainData = ImageFolder(path)
    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle = shuffle)
    return trainLoader