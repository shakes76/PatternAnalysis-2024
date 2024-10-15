from os.path import join
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
ADNI_DataLoader handles all data loaded to pass to the neural network
"""
class ADNI_DataLoader():
    rootDataPath: str
    imageSize: int
    dataTransforms : transforms.Compose
    
    __gotTrain : bool
    trainDataset : datasets.ImageFolder
    trainLoader : DataLoader
    
    __gotTest : bool
    testDataset : datasets.ImageFolder
    testLoader : DataLoader
    
    def __init__(self, rootData = "", imageSize = 240):
        self.rootDataPath = rootData
        self.imageSize = imageSize
        self.dataTransforms = self.__create_transforms__()
        self.__gotTrain = False
        self.__gotTest = False
        
    def __create_transforms__(self):
        all_transforms = transforms.Compose(
                [transforms.Resize(self.imageSize),
                transforms.CenterCrop(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)]) #TODO calculate mean and standard deviation of ADNI to use in normalization
        return all_transforms

    def __calc_mean_and_std(self): #TODO calculate mean & std here
        pass
    
    def get_dataloader(self, is_train: bool):
        if (is_train):
            if (not self.__gotTrain):
                path_to_dataset = join(self.rootDataPath, 'train')
                self.trainDataset = datasets.ImageFolder(path_to_dataset, transform=self.dataTransforms)
                self.trainLoader = DataLoader(self.trainDataset, batch_size=64, shuffle=True)
                self.__gotTrain = True
            return self.trainLoader
            
        else:
            if (not self.__gotTest):
                path_to_dataset = join(self.rootDataPath, 'test')
                self.testDataset = datasets.ImageFolder(path_to_dataset, transform=self.dataTransforms)
                self.testLoader = DataLoader(self.trainDataset, batch_size=64, shuffle=True)
                self.__gotTest = True
            return self.testLoader

