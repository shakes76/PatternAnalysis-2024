from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

ROOT = "../ADNI/AD_NC/train"
TESTPATH = "/test"
TRAINPATH = "/train"

def getTrainLoader(path = ROOT + TRAINPATH, batchSize = 128, shuffle = True):
    trainData = ImageFolder(path)
    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle = shuffle)
    return trainLoader

def getTestLoader(path = ROOT + TESTPATH, batchSize = 128, shuffle = True):
    trainData = ImageFolder(path)
    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle = shuffle)
    return trainLoader