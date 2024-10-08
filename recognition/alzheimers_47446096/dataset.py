from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

ROOT = "../../../AD_NC"
TESTPATH = "/test"
TRAINPATH = "/train"
IMG_SIZE = 256

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandAugment(num_ops = 3),
    transforms.ToTensor()
])
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor()
])

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
    trainData = ImageFolder(
        root = path,
        transform = TRAIN_TRANSFORM
    )
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
    trainData = ImageFolder(
        root = path,
        transform = TEST_TRANSFORM
    )
    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle = shuffle)
    return trainLoader