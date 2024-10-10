from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import shutil
from random import sample

ROOT = "../../../AD_NC"
NEW_ROOT = "../../../PatientSplit"
TESTPATH = "/test"
TRAINPATH = "/train"
VALPATH = "/val"
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

def getTrainLoader(path: str = NEW_ROOT + TRAINPATH, batchSize: int = 128, shuffle: bool = True):
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

def getValLoader(path = NEW_ROOT + VALPATH, batchSize = 128, shuffle = True):
    '''
    Get Pytorch DataLoader with ADNI test DATA

    Input:
        path: str - relative path to training data
        batchSize: int - batch size of the DataLoader
        suffle: bool - DataLoader shuffle option
    Returns:
        Pytorch DataLoader with ADNI test data loaded
    '''
    valData = ImageFolder(
        root = path,
        transform = TEST_TRANSFORM
    )
    valLoader = DataLoader(valData, batch_size = batchSize, shuffle = shuffle)
    return valLoader

def getTestLoader(path = NEW_ROOT + TESTPATH, batchSize = 128, shuffle = True):
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

def formatByPatient(path = ROOT, newPath = NEW_ROOT):
    if os.path.exists(newPath):
        print("Patient split has already been completed. Aborting:")
        return 1
    os.mkdir(newPath)
    print("Patient split has not been completed. Starting now:")
    q = []
    q.append((path, newPath))
    testroot = None
    while q:
        info = q.pop(0)
        p = info[0]
        newP = info[1]
        files = False
        for item in (os.listdir(p)):
            if (os.path.isfile(os.path.join(p, item))):
                files = True
                break
        if (files == True):
            for item in (os.listdir(p)):
                patient = item.split("_")[0]
                if not os.path.exists(os.path.join(newP, patient)):
                    os.mkdir(os.path.join(newP, patient))
                shutil.copy(os.path.join(p, item), os.path.join(newP, patient))
        else:
            for item in (os.listdir(p)):
                if item == "test":
                    testroot = (os.path.join(newP, item), os.path.join(newP, "val"))
                os.mkdir(os.path.join(newP, item))
                q.append((os.path.join(p, item), os.path.join(newP, item)))

    valq = []
    valq.append(testroot)
    os.mkdir(testroot[1])
    while valq:
        info = valq.pop(0)
        p = info[0]
        newP = info[1]
        files = False
        for item in (os.listdir(p)):
            item2 = os.listdir(os.path.join(p, item))[0]
            if (os.path.isfile(os.path.join(p, item, item2))):
                files = True
                break
        if files:
            samples = sample(os.listdir(p), (len(os.listdir(p))) // 2)
            for folder in samples:
                shutil.copytree(os.path.join(p, folder), os.path.join(newP, folder))
                shutil.rmtree(os.path.join(p, folder))
        else:
            for item in (os.listdir(p)):
                os.mkdir(os.path.join(newP, item))
                valq.append((os.path.join(p, item), os.path.join(newP, item)))
